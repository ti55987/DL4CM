import numpy as np
from utils.simulate_utils import generate_beta_with_diff_means_sim_vars

def generate_rlf_parameters(n_agents, default_params={}):
    default_params = {
        "bias": 1,
        "eps": 0,
    }    

    mean = 0.1
    var = 0.01  # Variance of both distributions
    phi_dist, _, _, _, _, _ = generate_beta_with_diff_means_sim_vars(
        mean, mean, var, n_agents, 0, 0.6
    )

    alpha_dist = np.random.uniform(0.2, 0.8, n_agents)
    # for testing agents, use smaller range of sticky_dist to align with real participants behavior
    sticky_dist = np.random.uniform(
        -1, 1, n_agents
    )  
    beta_dist = np.random.uniform(0.2, 0.8, n_agents)    

    return [
        {"alpha": alpha_dist[i],
        "phi": phi_dist[i],
        "stickiness": sticky_dist[i],
        "beta": beta_dist[i],
        "bias": default_params["bias"],
        "eps": default_params["eps"],
    } for i in range(n_agents)]   