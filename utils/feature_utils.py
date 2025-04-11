import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from tensorflow import one_hot

def get_labels(data, parameters):
    name_to_labels = {}
    for l in parameters:
      name_to_labels[l] = data.groupby('agentid')[l].agg(['mean']).to_numpy()

    return name_to_labels


def get_condition_aware_labels(data, shared_parameters, conditioned_parameters):  
  name_to_labels = get_labels(data, shared_parameters)

  grouped = data.groupby(['agentid', 'condition'])    
  # Extract unique values for each parameter in the list
  unique_values = grouped.apply(lambda x: pd.Series({
      param: x[param].unique()[0] for param in conditioned_parameters if len(x[param].unique()) == 1
  })).reset_index()

  num_conditions = data.condition.nunique()
  for i in range(num_conditions):
    cond_values = unique_values[unique_values.condition == i]
    for p in conditioned_parameters:
      name_to_labels[p+str(i)] = cond_values[p].to_numpy().reshape(-1, 1)
  
  return name_to_labels

# Recovery helper functions
def recover_parameter(prediction, scaler):
  estimated = prediction.reshape(prediction.shape[0], 1)
  return scaler.inverse_transform(estimated)[:, 0]

def get_recovered_parameters(name_to_scaler, name_to_true_parms, prediction):
  from collections import defaultdict

  sorted_label_names = list(name_to_true_parms.keys())
  sorted_label_names.sort()
  param_all_test = defaultdict(list)
  idx = 0
  for l in sorted_label_names:
    k = f'true_{l}'
    param_all_test[k] = name_to_true_parms[l][:, 0]

    k = f'dl_{l}'
    param_all_test[k] = recover_parameter(prediction[:, idx], name_to_scaler[l])
    idx += 1

  return pd.DataFrame(param_all_test)


def normalize_train_labels(name_to_labels: dict):
  names = list(name_to_labels.keys())
  names.sort()

  normalized_labels = []
  name_to_scaler = {}
  for name in names:
    scaler = StandardScaler()
    normalized_labels.append(scaler.fit_transform(name_to_labels[name]))
    name_to_scaler[name] = scaler

  return np.concatenate(normalized_labels, axis=-1), name_to_scaler

def normalize_val_labels(name_to_labels: dict, name_to_scaler: dict):
  names = list(name_to_labels.keys())
  names.sort()

  normalized_labels = []
  for name in names:
    scaler = name_to_scaler[name]
    normalized_labels.append(scaler.transform(name_to_labels[name]))

  return np.concatenate(normalized_labels, axis=-1)

def concat_blocks(f):
  n_a, n_b, n_t, n_d = f.shape
  return tf.reshape(f, [n_a, n_b*n_t, n_d])

def get_block_features(data):
  n_agent = len(data['agentid'].unique())
  n_trial = len(data['trials'].unique())
  n_block = len(data['block_no'].unique())
  action = data['actions'].to_numpy().astype(np.int32).reshape((n_agent, n_block, n_trial))
  reward = data['rewards'].to_numpy().astype(np.float64).reshape((n_agent, n_block, n_trial))

  n_action =  len(data['actions'].unique())
  action_onehot = to_categorical(action, n_action)

  return Concatenate(axis=3)([reward[:, :, :, np.newaxis], action_onehot]) #stimuli_onehot


def extract_features_blockless(data, input_list, onehot=False):
  from sklearn.preprocessing import OneHotEncoder

  agentids = data.agentid.unique()
  n_block = data.block_no.nunique()

  one_agent = data[data.agentid == agentids[0]]
  num_trials_per_agent = 0
  for b in range(n_block):
    num_trials_per_agent += one_agent[one_agent.block_no == b].trials.nunique()


  encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')  # sparse=False for numpy array

  n_agent = len(agentids)
  features = []
  for key in input_list:
    if key == 'actions' and onehot:
        # Reshape actions for one-hot encoding
        actions = data[key].to_numpy().astype(np.int64).reshape(-1, 1)
        # Fit and transform using OneHotEncoder
        encoded_actions = encoder.fit_transform(actions)
        # Reshape to (n_agent, num_trials_per_agent, n_actions)
        n_actions = encoded_actions.shape[1]
        one_hot_reshaped = encoded_actions.reshape((n_agent, num_trials_per_agent, n_actions))
        # Add each one-hot dimension as a separate feature
        for action_idx in range(n_actions):
            action_feature = one_hot_reshaped[:, :, action_idx]
            features.append(action_feature)
        #features.append(encoded_actions)
    else:
      f = data[key].to_numpy().astype(np.int64).reshape((n_agent, num_trials_per_agent))
      features.append(f)

  return np.stack(features, axis=-1)

def extract_features(data, input_list):
  n_agent = len(data['agentid'].unique())
  n_trial = len(data['trials'].unique())
  n_block = len(data['block_no'].unique())
  features = []
  for key in input_list:
    f = data[key].to_numpy().astype(np.int64).reshape((n_agent, n_block, n_trial))
    features.append(f)
  
  return np.stack(features, axis=-1)

def get_block_onehot_features(data, input_list):
  n_agent = len(data['agentid'].unique())
  n_trial = len(data['trials'].unique())
  n_block = len(data['block_no'].unique())
  features = []
  for key in input_list:
    input = data[key].to_numpy()
    unique_input = np.unique(input)

    cat_map = { item:i for i, item in enumerate(unique_input)}
    input_cat = [cat_map[s] for s in input]
    input_cat = np.array(input_cat).astype(np.int32).reshape((n_agent, n_block, n_trial))
    features.append(one_hot(input_cat, len(unique_input)))

  return np.concatenate(features, axis=-1)

def cut_block_data(block_data):
  """Cuts block_data into multiple arrays based on is_switch.

  Args:
    block_data: A pandas DataFrame containing block data.

  Returns:
    A list of arrays, where each array contains data from a single run.
  """

  # Get the indices of the switch trials.
  switch_indices = block_data.index[block_data['isswitch'] == 1].tolist()

  # Add the start and end indices of the block.
  switch_indices = [0] + switch_indices + [len(block_data)]

  # Create a list of arrays, where each array contains data from a single run.
  runs = []
  for i in range(len(switch_indices) - 1):
    start_index = switch_indices[i]
    end_index = switch_indices[i + 1]
    runs.append(block_data.iloc[start_index:end_index])

  return runs

def get_stimuli_iter_acc_vectorized(data):
    # Create a Series of stimulus counts

    # Initialize the count dictionary
    iteration_count = {}
    stim_iter_values = []
    # Calculate stim_iter values (this is the only part that needs iteration)
    for stim in data.stimuli:
        if stim not in iteration_count:
            iteration_count[stim] = 1
        else:
            iteration_count[stim] += 1
        stim_iter_values.append(iteration_count[stim])

    # Create a dictionary with all the data
    return {
        "stim_iter": stim_iter_values,
        "stimuli": data.stimuli.tolist(),
        "rewards": data.iscorrectaction.tolist(),
        "condition": data.condition.tolist(),
        "actions": data.actions.tolist(),
        "trials": data.trials.tolist(),
    }

def get_iter_acc_without_switches(train_df):
    """
    Process training data to calculate stimulus iteration accuracy.

    Args:
        train_df: DataFrame containing training data

    Returns:
        DataFrame with processed iteration accuracy data
    """
    # Process the data using DataFrames for each block
    result_dfs = []

    for (agent_id, block_no), block_data in train_df.groupby(['agentid', 'block_no']):
        # Reset index for consistent processing
        block_data = block_data.reset_index(drop=True)

        # Process this block's data
        block_results = get_stimuli_iter_acc_vectorized(block_data)

        # Skip empty results
        if not block_results or not any(len(v) > 0 for v in block_results.values()):
            continue

        # Convert to DataFrame and add metadata
        block_df = pd.DataFrame(block_results)
        block_df['agentid'] = agent_id
        block_df['block_no'] = block_no

        # Get set_size - assuming it's constant within a block
        if 'set_size' in block_df.columns and len(block_data) > 0:
            block_df['set_size'] = block_data.set_size.iloc[0]

        # Add to results
        result_dfs.append(block_df)

    # Combine all results
    return pd.concat(result_dfs, ignore_index=True) if result_dfs else pd.DataFrame()


def get_iter_acc_with_switches(train_df):
    iter_acc = {
        "stim_iter": [], "rewards": [], "block_no": [], "condition": [],
        "agentid": [], "isswitch": [], "set_size": [], "trials": [],
        "actions": [], "stimuli": [],
    }

    # Group data by agentid and block_no to avoid repeated filtering
    for (agent_id, block_no), block_data in train_df.groupby(['agentid', 'block_no']):
        block_data = block_data.reset_index(drop=True)  # Reset index once per group

        for run_data in cut_block_data(block_data):
            block_results = get_stimuli_iter_acc_vectorized(run_data)

            # Extend lists with computed results
            for key in ["stim_iter", "rewards", "condition", "stimuli", "actions"]:
                iter_acc[key].extend(block_results[key])

            for key in ["trials", "isswitch"]:
                iter_acc[key].extend(run_data[key].tolist())

        # Extend agent, block, and set_size info
        block_size = len(block_data)
        iter_acc["agentid"].extend([agent_id] * block_size)
        iter_acc["block_no"].extend([block_no] * block_size)
        iter_acc["set_size"].extend(
            block_data["set_size"].tolist() if "set_size" in block_data.columns else [6] * block_size
        )

    return pd.DataFrame(iter_acc)

