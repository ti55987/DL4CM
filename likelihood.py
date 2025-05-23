import numpy as np
import random

import scipy
from scipy.optimize import minimize


BETA_MULTIPLIER = 20
# 2PRL likelihood
def prl2_neg_log_likelihood(data, parameters):
  alpha, beta = parameters
  beta = beta*BETA_MULTIPLIER # why do it here?

  #print(alpha, beta)
  num_actions = len(data.actions.unique())
  num_stimuli = len(data.stimuli.unique())
  q_values = np.array([1/num_actions]*num_actions) # equal value first
  llh = 0
  for a, r in zip(data.actions, data.rewards):
    llh += np.log(scipy.special.softmax(beta * q_values)[a])

    rpe = r - q_values[a]
    q_values[a] += alpha*rpe # update q value

    unchosen_rpe = (1-r) - q_values[1-a]
    q_values[1-a] += alpha*unchosen_rpe # update q value
  return -llh

# 2PRL-SA likelihood
def rl_sa_neg_log_likelihood(data, parameters):
  if len(parameters) == 2:
    alpha0, beta = parameters
    alpha1 = alpha0
    phi=0
  elif len(parameters) == 3:
    alpha0, beta, sticky = parameters
    alpha1 = alpha0
    phi = 0
  elif len(parameters) == 4:
    alpha0, beta, sticky, phi = parameters
    alpha1 = alpha0
    #alpha0, alpha1, beta = parameters

  beta = beta*BETA_MULTIPLIER

  num_actions = len(data.actions.unique())
  llh = 0
  for b in data.block_no.unique():
    block_data = data[data.block_no == b]
    condition = block_data.condition.iloc[0]
    alpha = alpha0 if condition == 0 else alpha1


    num_stimuli = len(block_data.stimuli.unique())
    init_value = 1.0 / num_actions
    q_values = {i:  np.array([init_value]*num_actions) for i in range(num_stimuli)} # equal value first
    prev_a = -1
    for s, a, r in zip(block_data.stimuli, block_data.actions, block_data.rewards):
      Q = q_values.copy()
      if prev_a != -1:
        Q[s][prev_a] = Q[s][prev_a]+sticky

      llh += np.log(scipy.special.softmax(beta * Q[s])[a])
      prev_a = a      
      #llh += np.log(scipy.special.softmax(beta * q_values[s])[a])

      # Forgetting - fix to case with different Q/W
      for st, action_to_prob in q_values.items():
          for i in range(len(action_to_prob)):
              # same thing as WM = WM + forget (1/n - WM)
              q_values[st][i] = (1.0 - phi) * q_values[st][i] + phi * init_value

      rpe = r - q_values[s][a]
      q_values[s][a] += alpha*rpe # update q value
      for x in list(np.arange(num_actions)):
        if x == a:
          continue
        #RPE for the unselected action
        rpe_unchosen = (1-r)-q_values[s][x]
        q_values[s][x] += (alpha*rpe_unchosen)

  return -llh

def prl4_neg_log_likelihood(actions, rewards, parameters):
  alpha, neg_alpha, beta, stickiness = parameters

  beta = beta*BETA_MULTIPLIER
  num_actions = 2

  lr_list = [neg_alpha, alpha]
  q_values = np.array([1/num_actions]*num_actions) # equal value first

  llh = 0
  prev_a = -1
  for a, r in zip(actions, rewards):
    Q = q_values.copy()
    if prev_a != -1:
       Q[prev_a] = Q[prev_a]+stickiness

    llh += np.log(scipy.special.softmax(beta * Q)[a])

    rpe = r - q_values[a]
    q_values[a] += lr_list[r]*rpe # update q value

    unchosen_rpe = (1-r) - q_values[1-a]
    q_values[1-a] += lr_list[r]*unchosen_rpe # update q value
    prev_a = a

  return -llh

# Function to process a single agent ID
def process_agent(aid, data, metadata, bound_name='bounds', max_iterations=30, likelihood_func=rl_sa_neg_log_likelihood):
    """Process a single agent ID and return the optimization results."""
    try:
        print(f'Starting optimization for agent {aid}...')
        sub_data = data[data.agentid == aid]
        init_params = [random.uniform(l, h) for l, h in metadata[bound_name]]

        # Define the function for this agent
        func = lambda x, *args: likelihood_func(sub_data, x)

        # Run optimization
        res = minimize(
            func,
            init_params,
            bounds=metadata[bound_name],
            method='L-BFGS-B',
            options={'maxiter': max_iterations}
        )

        print(f'Completed optimization for agent {aid}')
        return aid, res.x
    except Exception as e:
        print(f'Error processing agent {aid}: {str(e)}')
        return aid, None