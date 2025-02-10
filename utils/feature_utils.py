import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler


def get_labels(data, parameters):
    name_to_labels = {}
    for l in parameters:
      name_to_labels[l] = data.groupby('agentid')[l].agg(['mean']).to_numpy()

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

def get_block_features(data):
  n_agent = len(data['agentid'].unique())
  n_trial = len(data['trials'].unique())
  n_block = len(data['block_no'].unique())
  action = data['actions'].to_numpy().astype(np.int32).reshape((n_agent, n_block, n_trial))
  reward = data['rewards'].to_numpy().astype(np.float64).reshape((n_agent, n_block, n_trial))

  n_action =  len(data['actions'].unique())
  action_onehot = to_categorical(action, n_action)

  return Concatenate(axis=3)([reward[:, :, :, np.newaxis], action_onehot]) #stimuli_onehot