import random
import numpy as np
import pandas as pd
from rl_models import PRL
from mixture_models import WMMixture
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


def get_last_stimuli_trial(df):
    # Create a copy of the dataframe with trials and stimuli
    temp_df = df[["trials", "stimuli"]].copy()
    # Group by stimuli and shift trials to get previous trial number
    temp_df["last_trial"] = temp_df.groupby("stimuli")["trials"].shift(1)
    return temp_df["last_trial"].values


def simulate_agent(
    a,
    pval,
    num_blocks,
    num_stimuli_list,
    min_switches,
    iter_per_stimuli,
    num_actions,
    all_seq,
    params_dist={},
):

    agent = PRL(
        beta=params_dist["beta"] * 20,
        pval=pval,
        id=a,
        phi=params_dist["phi"],
        stickiness=params_dist["stickiness"],
        bias=params_dist["bias"],
        eps=params_dist["eps"],
    )

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

        agent.init_model(
            alpha=params_dist["alpha"],
            stimuli=np.arange(num_stimuli),
            actions=np.arange(num_actions),
            mapping=mappings,
        )
        data = agent.simulate_block(
            num_trials=num_trials, stimuli=all_seq[block_no], min_switch=min_switch
        )
        # data['last_stimuli_trial'] = get_last_stimuli_trial(data)
        data["delay_since_last_stimuli"] = data["trials"] - pd.Series(
            get_last_stimuli_trial(data)
        )
        data["delay_since_last_stimuli"] = (
            data["delay_since_last_stimuli"].fillna(0).astype(int)
        )
        # data["delay_since_last_correct"] = data["trials"] - pd.Series(
        #     get_last_correct_trial(data)
        # )
        # data["delay_since_last_correct"] = (
        #     data["delay_since_last_correct"].fillna(0).astype(int)
        # )
        data["block_no"] = [block_no] * num_trials
        data["condition"] = [cond] * num_trials
        data["set_size"] = [num_stimuli] * num_trials
        agent_data_list.append(data)

    return pd.concat(agent_data_list)


def simulate_mixture_agent(
    a,
    num_blocks,
    num_stimuli_list,
    iter_per_stimuli,
    num_actions,
    all_seq,
    using_rl=False,
    params_dist={},
):

    agent = WMMixture(
        id=a,
        eta6_wm=params_dist["eta6_wm"],
        r0=0 if using_rl else 1,
        phi=params_dist["phi"],
        stickiness=params_dist["stickiness"],
        bias=params_dist["bias"],
        eps=params_dist["eps"],
    )

    half_block_no = int(num_blocks / 2)
    conditions = [0] * half_block_no + [1] * half_block_no
    random.shuffle(conditions)
    agent_data_list = []

    for block_no in range(num_blocks):
        cond = conditions[block_no]
        num_stimuli = num_stimuli_list[cond]
        mappings = generate_valid_mappings(num_stimuli, num_actions)
        num_trials = num_stimuli * iter_per_stimuli

        agent.init_model(
            learning_rate=params_dist["alpha"],
            stimuli=np.arange(num_stimuli),
            actions=np.arange(num_actions),
            mapping=mappings,
        )
        data = agent.simulate_block(
            num_trials=num_trials, stimuli=all_seq[block_no], sz=num_stimuli
        )
        # data['last_stimuli_trial'] = get_last_stimuli_trial(data)
        data["delay_since_last_stimuli"] = data["trials"] - pd.Series(
            get_last_stimuli_trial(data)
        )
        data["delay_since_last_stimuli"] = (
            data["delay_since_last_stimuli"].fillna(0).astype(int)
        )

        data["block_no"] = [block_no] * num_trials
        data["condition"] = [cond] * num_trials
        data["set_size"] = [num_stimuli] * num_trials
        agent_data_list.append(data)

    return pd.concat(agent_data_list)