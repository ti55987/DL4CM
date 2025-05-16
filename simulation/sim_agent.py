import random
import numpy as np
import pandas as pd
import tqdm
from concurrent.futures import ProcessPoolExecutor
from utils.simulate_utils import generate_beta_with_diff_means_sim_vars
from rl_models import PRL
from utils.simulate_utils import generate_valid_mappings


def get_last_correct_trial(df):
    correct_trial_map = {}
    last_correct_trial = []
    for _, row in df.iterrows():
        s = row.stimuli
        if s in correct_trial_map:
            last_correct_trial.append(correct_trial_map[s])
        else:
            last_correct_trial.append(None)

        if row.rewards == 1:
            correct_trial_map[s] = int(row.trials)
    
    return last_correct_trial

def simulate_agent(a, phi_dist, pval, num_blocks, num_stimuli_list, min_switches, iter_per_stimuli, num_actions, all_seq):
    rand_beta = random.uniform(0.2, 0.8)
    rand_alpha = random.uniform(0.2, 0.8)
    rand_phi = phi_dist[a]
    agent = PRL(beta=rand_beta*20, pval=pval, id=a, phi=rand_phi)

    half_block_no = int(num_blocks / 2)
    conditions = [0] * half_block_no + [1] * half_block_no
    random.shuffle(conditions)
    agent_data_list = []

    for block_no in range(num_blocks):
        cond = conditions[block_no]
        num_stimuli = num_stimuli_list[cond]
        min_switch = min_switches[cond]
        mappings = generate_valid_mappings(num_stimuli, num_actions)
        num_trials = num_stimuli * iter_per_stimuli

        agent.init_model(alpha=rand_alpha, stimuli=np.arange(num_stimuli), actions=np.arange(num_actions), mapping=mappings)
        data = agent.simulate_block(num_trials=num_trials, stimuli=all_seq[block_no], min_switch=min_switch)
        data['last_correct_trial'] = get_last_correct_trial(data)
        data['delay_since_last_correct'] = data['trials'] - data['last_correct_trial']
        data['delay_since_last_correct'] = data['delay_since_last_correct'].fillna(0).astype(int)
        data['block_no'] = [block_no] * num_trials
        data['condition'] = [cond] * num_trials
        data['set_size'] = [num_stimuli] * num_trials
        agent_data_list.append(data)

    return pd.concat(agent_data_list)

