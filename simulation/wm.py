import numpy as np
from utils.simulate_utils import generate_beta_with_diff_means_sim_vars


def generate_wm_parameters(n_agents, default_params={}):
    default_params = {
        "alpha": 1,
        "beta": 1.25,
    }

    mean = 0.1
    var = 0.01  # Variance of both distributions
    phi_dist, _, _, _, _, _ = generate_beta_with_diff_means_sim_vars(
        mean, mean, var, n_agents, 0, 0.6
    )
    eps_dist, _, _, _, _, _ = generate_beta_with_diff_means_sim_vars(
        mean, mean, var, n_agents, 0, 1
    )
    sticky_dist = np.random.uniform(-1, 1, n_agents)
    bias_dist = np.random.uniform(0, 1, n_agents)
    return [{
        "alpha": default_params["alpha"],
        "phi": phi_dist[i],
        "stickiness": sticky_dist[i],
        "beta": default_params["beta"],
        "bias": bias_dist[i],
        "eps": eps_dist[i],
    } for i in range(n_agents)]
