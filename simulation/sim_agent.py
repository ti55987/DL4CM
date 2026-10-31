import random
import numpy as np
import pandas as pd
from rl_models import PRL
from utils.simulate_utils import generate_valid_mappings
from utils.data_utils import get_last_stimuli_trial


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
    cond_alpha = {
        0: params_dist["alpha"] if "alpha" in params_dist else params_dist["alpha0"],
        1: params_dist["alpha"] if "alpha" in params_dist else params_dist["alpha1"],
    }
    agent_data_list = []
    for block_no in range(num_blocks):
        cond = conditions[block_no]
        num_stimuli = num_stimuli_list[cond]
        min_switch = min_switches[cond]
        mappings = generate_valid_mappings(num_stimuli, num_actions)
        num_trials = num_stimuli * iter_per_stimuli

        agent.init_model(
            alpha=cond_alpha[cond],
            neg_alpha=(
                params_dist["neg_alpha"]
                if "neg_alpha" in params_dist
                else cond_alpha[cond]
            ),
            stimuli=np.arange(num_stimuli),
            actions=np.arange(num_actions),
            mapping=mappings,
        )
        data = agent.simulate_block(
            num_trials=num_trials, stimuli=all_seq[block_no], min_switch=min_switch
        )
   
        data["delay_since_last_stimuli"] = data["trials"] - pd.Series(
            get_last_stimuli_trial(data)
        )
        data["delay_since_last_stimuli"] = (
            data["delay_since_last_stimuli"].fillna(0).astype(int)
        )
        data["block_no"] = [block_no] * num_trials
        data["condition"] = [cond] * num_trials
        data["set_size"] = [num_stimuli] * num_trials
        if 'alpha0' in params_dist:
            data["alpha0"] = [params_dist["alpha0"]] * num_trials
        if 'alpha1' in params_dist:
            data["alpha1"] = [params_dist["alpha1"]] * num_trials
        
        agent_data_list.append(data)

    return pd.concat(agent_data_list)


