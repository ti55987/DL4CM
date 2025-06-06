from ast import Return
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

import logging
import time
import math
import os
import h5py
import plotnine as gg

from scipy import stats
from datetime import datetime
from packaging import version
from enum import Enum
from tensorflow import one_hot
from tensorflow.python.keras.layers import Concatenate

class Mode(Enum):
    PRL2 = 1
    PRL4 = 2
    PRL2_intractable = 3
    HRL2 = 4
    HRL2_fixed_beta = 5
    HRL2_Bayes = 6
    HRL2_StickyBayes = 7
    Bayes = 8
    StickyBayes = 9
    PRL2_intractable_high_tau = 10

LIST_HRL_MODELS = [Mode.HRL2, Mode.HRL2_fixed_beta, Mode.HRL2_Bayes, Mode.HRL2_StickyBayes]

def read_hdf5(path):
    data = {}
    with h5py.File(path, 'r') as f: # open file
      k = f['df']['block0_items'][()]
      values = f['df']['block0_values'][()]
      for idx in range(len(k)):
        cleaned_key = str(k[idx]).replace("b'", '').rstrip("'")
        data[cleaned_key] = values[:, idx]

      k = f['df']['block1_items'][()]
      values = f['df']['block1_values'][()]
      for idx in range(len(k)):
        cleaned_key = str(k[idx]).replace("b'", '').rstrip("'")
        data[cleaned_key] = values[:, idx]
    return pd.DataFrame(data)

def get_features(data, n_agent, n_trial, n_action=2, mode=Mode.PRL2):
    if mode in LIST_HRL_MODELS:
      stim_prefix = 'stim' if mode == Mode.HRL2_fixed_beta else 'allstims'

      side0=data[f'{stim_prefix}0'].to_numpy().astype(np.float32).reshape((n_agent, n_trial))
      side1=data[f'{stim_prefix}1'].to_numpy().astype(np.float32).reshape((n_agent, n_trial))
      side2=data[f'{stim_prefix}2'].to_numpy().astype(np.float32).reshape((n_agent, n_trial))
      action = data['chosenside'].to_numpy().astype(np.int32).reshape((n_agent, n_trial))
      reward = data['rewards'].to_numpy().astype(np.float32).reshape((n_agent, n_trial))
      # turn action into one-hot
      action_onehot = one_hot(action, n_action)
      # concatenate reward with action
      return Concatenate(axis=2)(
        [side0[:, :, np.newaxis],side1[:, :, np.newaxis],side2[:, :, np.newaxis],reward[:, :, np.newaxis], action_onehot])
    
    
    # Probablistic RL
    action = data['actions'].to_numpy().astype(np.int32).reshape((n_agent, n_trial))
    reward = data['rewards'].to_numpy().astype(np.float32).reshape((n_agent, n_trial))

    # turn action into one-hot
    action_onehot = one_hot(action, n_action)
    # concatenate reward with action
    return Concatenate(axis=2)([reward[:, :, np.newaxis], action_onehot])      

def concat_by_cols(a, b):
  return np.hstack([a, b]) if a.size else b

def padding(data,  max_num_trial: int, num_trials: int):
  paddings = tf.constant([[0, max_num_trial-num_trials], [0, 0]])
  a_list = tf.unstack(data)
  for j in range(len(data)):
    padded_inputs = tf.pad(data[j], paddings, "CONSTANT", constant_values=-1)
    a_list[j] = padded_inputs

  return tf.stack(a_list)
  
def get_label_names_by_mode(mode):
    if mode == Mode.PRL2 or mode == Mode.HRL2:
      return ['alpha', 'beta']
    elif mode == Mode.PRL2_intractable_high_tau:
      return ['alpha', 'beta', 'T']
    elif mode == Mode.PRL2_intractable:
      return ['alpha', 'beta', 'to_inattentive_tau']
    elif mode == Mode.HRL2_fixed_beta:
      return ['alpha', 'stickiness']
    elif mode == Mode.Bayes:
      return ['beta','preward','pswitch']
    elif mode == Mode.StickyBayes:
      return ['beta','preward','pswitch','stickiness']
    elif mode  == Mode.HRL2_Bayes:
      return ['beta','epsilon']
    elif mode == Mode.HRL2_StickyBayes:
      return ['beta','epsilon','stickiness']
    else:
      return ['alpha', 'beta', 'neg_alpha', 'stickiness']

def get_columns_by_label(label: str):
  if label == 'beta':
    return 'TrueBeta', 'MAPBeta'
  elif label == 'alpha':
    return 'TrueAlpha', 'MAPAlpha'
  elif label == 'neg_alpha':
    return 'TrueAlphaNeg', 'MAPAlphaNeg'
  elif label == 'preward':
    return 'Truepreward','MAPpreward'
  elif label == 'pswitch':
    return 'Truepswitch','MAPpswitch'
  elif label == 'stickiness':
     return 'Truestickiness','MAPstickiness'
  elif label == 'epsilon':
    return 'TrueEpsilon','MAPEpsilon'
  else:
    return 'TrueStick', 'MAPStick'

def plot_loss(benchmark_file: str, history, loss_file: str, title: str=''):
  x_length = len(history.history['loss'])
  plt.plot(history.history['loss'], label = 'training')
  plt.plot(history.history['val_loss'], label = 'validation')
  plt.xlabel('epoch')
  plt.ylabel('loss')

  # MAP=pd.read_csv(benchmark_file)
  # predicted_labels = get_predicted_labels_by_mode(mode)
  # sumMSE_MAP, errorbarMSE = 0, 0
  # for l in predicted_labels:
  #   sumMSE_MAP = sumMSE_MAP + get_map_sum_by_label(MAP, l)
  #   errorbarMSE = errorbarMSE + get_map_std_by_label(MAP, l)
  # plt.hlines(sumMSE_MAP+errorbarMSE, 0, x_length, colors='gray')
  # plt.hlines(sumMSE_MAP-errorbarMSE, 0, x_length, colors='gray')
  # plt.hlines(sumMSE_MAP, 0, x_length, colors='k', linestyles='dashed', label='MAP Loss Benchmark')
  plt.title(title)
  leg = plt.legend(loc='upper right')
  if len(loss_file) > 0:
    plt.savefig(loss_file)
  plt.show()
  
# ==================================
# Loss helper functions
def get_mse_label(param_all_test: dict, label :str):
  true_label, dl_label = f'true_{label}', f'dl_{label}' 
  y_true = param_all_test[true_label]
  y_pred = param_all_test[dl_label]
  mse = tf.keras.losses.MeanSquaredError()
  return mse(y_true, y_pred).numpy()

def get_dl_std_by_label(param_all_test: dict, l :str):
  true_label, dl_label = f'true_{l}', f'dl_{l}'
  return np.std((param_all_test[true_label]-param_all_test[dl_label]))/np.sqrt(len(param_all_test[true_label]))

def get_dl_sum_by_label(param_all_test: dict, label :str):
  true_label, dl_label = f'true_{label}', f'dl_{label}' 
  return np.sum((param_all_test[true_label]-param_all_test[dl_label])**2)/len(param_all_test[true_label])

def get_map_std_by_label(MAP: dict, label :str):
  true_label, map_label = get_columns_by_label(label)
  return np.std((MAP[true_label]-MAP[map_label]))/np.sqrt(len(MAP[true_label]))

def get_map_sum_by_label(MAP: dict, label :str):
  true_label, map_label = get_columns_by_label(label)
  return np.sum((MAP[true_label]-MAP[map_label])**2)/len(MAP[true_label])

# ==================================
def plot_recovery(param_all, label):
  from scipy.stats import spearmanr

  true_l, dl_l = f'true_{label}', f'dl_{label}'       

  r_value, p_value = spearmanr(param_all[true_l], param_all[dl_l])
  r_value = round(r_value, 2)
  p_value = round(p_value, 2)
  annotated_xp = param_all[true_l].mean()
  annotated_yp = param_all[dl_l].min()
  
  return (gg.ggplot(param_all, gg.aes(x = true_l, y = dl_l))
  + gg.geom_point(color = 'blue')
  + gg.stat_smooth(method = 'lm')
  + gg.geom_line(param_all, gg.aes(x = true_l, y = true_l), color="red", size = 1.2)
  + gg.annotate('label', x=annotated_xp, y=annotated_yp, label=f'R={r_value}, p={p_value}', size=9, color='#252525',
            label_size=0, fontstyle='italic')  
  ) 
