import numpy as np
import random

import scipy
from scipy.optimize import minimize
from utils.stats_utils import calculate_aic, calculate_bic

BETA_MULTIPLIER = 20


# 2PRL likelihood
def prl2_neg_log_likelihood(data, parameters):
    alpha, beta = parameters
    beta = beta * BETA_MULTIPLIER  # why do it here?

    # print(alpha, beta)
    num_actions = len(data.actions.unique())
    q_values = np.array([1 / num_actions] * num_actions)  # equal value first
    llh = 0
    for a, r in zip(data.actions, data.rewards):
        llh += np.log(scipy.special.softmax(beta * q_values)[a])

        rpe = r - q_values[a]
        q_values[a] += alpha * rpe  # update q value

        unchosen_rpe = (1 - r) - q_values[1 - a]
        q_values[1 - a] += alpha * unchosen_rpe  # update q value
    return -llh


def rl2_sa_neg_log_likelihood(data, param_dict):
    param_dict = {
        "alpha": param_dict["alpha"],
        "beta": param_dict["beta"],
        "stickiness": 0,
        "phi": 0,
        "bias": 1,
    }
    return sa_neg_log_likelihood_v2(data, param_dict)


def wm3_sa_neg_log_likelihood(data, parameters):
    beta, sticky, phi = parameters
    param_dict = {"alpha": 1, "beta": beta, "stickiness": sticky, "phi": phi, "bias": 1}
    return sa_neg_log_likelihood_v2(data, param_dict)


def rl4_sa_neg_log_likelihood(data, param_dict):
    param_dict = {
        "alpha": param_dict["alpha"],
        "beta": param_dict["beta"],
        "stickiness": param_dict["stickiness"],
        "neg_alpha": param_dict["neg_alpha"],
        "phi": 0,
        "bias": 1,
    }
    return sa_neg_log_likelihood_v2(data, param_dict)


# 2PRL-SA likelihood
def sa_neg_log_likelihood_v2(data, param_dict):
    from rl_models import PRL

    alpha = param_dict["alpha"] if "alpha" in param_dict else 1
    neg_alpha = param_dict["neg_alpha"] if "neg_alpha" in param_dict else alpha
    alpha_cond = {
        0: alpha,
        1: alpha,
    }

    num_actions = len(data.actions.unique())
    num_stimuli = len(data.stimuli.unique())
    agent = PRL(
        beta=param_dict["beta"] * BETA_MULTIPLIER if "beta" in param_dict else 25,
        pval=1,
        id=0,
        phi=param_dict["phi"],
        stickiness=param_dict["stickiness"],
        bias=param_dict["bias"],
        eps=param_dict["eps"] if "eps" in param_dict else 0,
    )
    llh = 0
    for b in data.block_no.unique():
        block_data = data[data.block_no == b]
        condition = block_data.condition.iloc[0]
        agent.init_model(
            alpha=alpha_cond[condition],
            neg_alpha=neg_alpha,
            stimuli=np.arange(num_stimuli),
            actions=np.arange(num_actions),
            mapping={},
        )
        prev_a = -1
        for s, a, r in zip(block_data.stimuli, block_data.actions, block_data.rewards):
            llh += np.log(agent.get_policy(s, prev_a)[a])
            prev_a = a
            agent.update_values(s, a, r)

    return -llh


def sa_mixture_neg_log_likelihood(data, param_dict):
    from mixture_models import create_mixture_model

    alpha = param_dict["alpha"] if "alpha" in param_dict else 1
    alpha_cond = {
        0: alpha,
        1: alpha,
    }

    num_actions = len(data.actions.unique())
    agent = create_mixture_model(id=0, params_dist=param_dict, using_rl=False)

    llh = 0
    for b in data.block_no.unique():
        block_data = data[data.block_no == b]
        num_stimuli = block_data.stimuli.nunique()
        condition = block_data.condition.iloc[0]
        agent.init_model(
            learning_rate=alpha_cond[condition],
            stimuli=np.arange(num_stimuli),
            actions=np.arange(num_actions),
            mapping={},
        )
        llh += agent.neg_log_likelihood(
            block_data.stimuli, block_data.actions, block_data.rewards, num_stimuli
        )

    return llh


def prl4_neg_log_likelihood(actions, rewards, parameters):
    alpha, neg_alpha, beta, stickiness = parameters

    beta = beta * BETA_MULTIPLIER
    num_actions = 2

    lr_list = [neg_alpha, alpha]
    q_values = np.array([1 / num_actions] * num_actions)  # equal value first

    llh = 0
    prev_a = -1
    for a, r in zip(actions, rewards):
        Q = q_values.copy()
        if prev_a != -1:
            Q[prev_a] = Q[prev_a] + stickiness

        llh += np.log(scipy.special.softmax(beta * Q)[a])

        rpe = r - q_values[a]
        q_values[a] += lr_list[r] * rpe  # update q value

        unchosen_rpe = (1 - r) - q_values[1 - a]
        q_values[1 - a] += lr_list[r] * unchosen_rpe  # update q value
        prev_a = a

    return -llh


class UniformPrior:
    """Uniform prior distribution for a parameter.

    Args:
        lower: Lower bound of the uniform distribution
        upper: Upper bound of the uniform distribution
    """

    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper
        self.log_prob = -np.log(upper - lower)  # log of 1/(upper-lower)
        self.min_log_prob = -1e10  # large negative number instead of -inf

    def __call__(self, x):
        """Compute log probability of uniform distribution.

        Args:
            x: Parameter value

        Returns:
            Log probability of uniform distribution
        """
        if self.lower <= x <= self.upper:
            return np.clip(self.log_prob, self.min_log_prob, 0)
        return self.min_log_prob  # use large negative number instead of -inf


def uniform_prior(lower, upper):
    """Creates a uniform prior for a parameter.

    Args:
        lower: Lower bound of the uniform distribution
        upper: Upper bound of the uniform distribution

    Returns:
        UniformPrior instance
    """
    return UniformPrior(lower, upper)


class BetaPrior:
    """Beta prior distribution for a parameter.

    Args:
        alpha: First shape parameter of beta distribution
        beta: Second shape parameter of beta distribution
        lower: Lower bound for scaling (default: 0)
        upper: Upper bound for scaling (default: 1)
    """

    def __init__(self, alpha, beta, lower=0, upper=1):
        from scipy.stats import beta as beta_dist

        self.alpha = alpha
        self.beta = beta
        self.lower = lower
        self.upper = upper
        self.beta_dist = beta_dist
        self.scale_factor = upper - lower
        self.min_log_prob = -1e10  # large negative number instead of -inf

        # Add small epsilon to bounds to handle numerical precision
        self.epsilon = 1e-10
        self.lower_with_epsilon = lower - self.epsilon
        self.upper_with_epsilon = upper + self.epsilon

    def __call__(self, x):
        """Compute log probability of beta distribution.

        Args:
            x: Parameter value

        Returns:
            Log probability of beta distribution
        """
        # Use epsilon-adjusted bounds for numerical stability
        if self.lower_with_epsilon <= x <= self.upper_with_epsilon:
            # Scale x to [0,1] interval with numerical stability
            scaled_x = np.clip((x - self.lower) / self.scale_factor, 0, 1)
            # Compute log probability of beta distribution
            log_prob = self.beta_dist.logpdf(scaled_x, self.alpha, self.beta) - np.log(
                self.scale_factor
            )
            # Handle potential numerical issues
            return np.clip(log_prob, self.min_log_prob, 0)
        return self.min_log_prob  # use large negative number instead of -inf


def beta_prior(alpha, beta, lower=0, upper=1):
    """Creates a beta prior for a parameter.

    Args:
        alpha: First shape parameter of beta distribution
        beta: Second shape parameter of beta distribution
        lower: Lower bound for scaling (default: 0)
        upper: Upper bound for scaling (default: 1)

    Returns:
        BetaPrior instance
    """
    return BetaPrior(alpha, beta, lower, upper)


def get_free_parameters(param_bounds_dict):
    param_names = [k for k, v in param_bounds_dict.items() if v[0] != v[1]]
    return param_names


# Function to process a single agent ID
def process_agent(
    aid,
    data,
    metadata,
    param_bounds_dict,
    max_iterations=30,
):
    """Process a single agent ID and return the optimization results."""
    likelihood_func = metadata["likelihood_func"]
    # Get parameter names and bounds
    param_names = sorted(list(param_bounds_dict.keys()))
    bounds = [param_bounds_dict[param] for param in param_names]
    sub_data = data[data.agentid == aid]

    # Create a wrapper function that unpacks dictionary to list for the likelihood function
    def func(params_list, *args):
        # Convert params list back to dictionary for tracking/debugging
        params_dict = {name: value for name, value in zip(param_names, params_list)}
        params_dict["r0"] = metadata["r0"] if "r0" in metadata else 0
        return likelihood_func(sub_data, params_dict)

    try:
        print(f"Starting optimization for agent {aid}...")

        init_params = [random.uniform(l, h) for l, h in bounds]
        # Run optimization
        res = minimize(
            func,
            init_params,
            bounds=bounds,
            method="L-BFGS-B",
            options={"maxiter": max_iterations},
        )

        # Calculate fit metrics
        llh = res.fun
        n_data_points = len(sub_data)
        n_params = len(get_free_parameters(param_bounds_dict))
        # Calculate AIC and BIC
        aic = calculate_aic(n_params, -llh)  # 2 * n_params + 2 * best_res.fun
        bic = calculate_bic(n_data_points, n_params, -llh)

        print(f"AIC for {aid}: {aic}")
        print(f"BIC for {aid}: {bic}")
        # Prepare result dictionary
        result = {
            "id": aid,
            "data_model": metadata["data_model"],  # The model that generated this data
            "fit_model": metadata[
                "model_name"
            ],  # The model settings used to fit the data
            "llh": llh,
            "aic": aic,
            "bic": bic,
            "params": res.x,
            "param_names": sorted(list(param_bounds_dict.keys())),
        }
        return result
    except Exception as e:
        print(f"Error processing agent {aid}: {str(e)}")
        return {"id": aid, "error": str(e)}


def process_agent_map(
    aid,
    data,
    metadata,
    param_bounds_dict,
    max_iterations=30,
):
    """Process a single agent ID and return the MAP optimization results.

    Args:
        aid: Agent ID
        data: DataFrame containing the data
        metadata: Dictionary containing bounds and other metadata
        bound_name: Name of the bounds in metadata
        max_iterations: Maximum number of optimization iterations
        likelihood_func: Function to compute negative log likelihood

    Returns:
        Tuple of (agent_id, optimized_parameters)
    """
    # Get parameter names and bounds
    param_names = sorted(list(param_bounds_dict.keys()))
    bounds = [param_bounds_dict[param] for param in param_names]

    # Create a wrapper function that unpacks dictionary to list for the likelihood function
    def likelihood_func(d, params, *args):
        f = metadata["likelihood_func"]
        # Convert params list back to dictionary for tracking/debugging
        params_dict = {name: value for name, value in zip(param_names, params)}
        params_dict["r0"] = metadata["r0"] if "r0" in metadata else 0
        return f(d, params_dict)

    # Define the uniform prior log probability function
    def log_prior(params):
        log_prob = 0.0
        pf = metadata["prior_func"]
        for name, value in zip(param_names, params):
            log_prob += pf[name](value)
        return log_prob

    try:
        print(f"Starting MAP optimization for agent {aid}...")
        init_params = [random.uniform(l, h) for l, h in bounds]
        sub_data = data[data.agentid == aid]

        # Define the function for this agent (negative log posterior)
        def neg_log_posterior(x, *args):
            # Negative log posterior = negative log likelihood - log prior
            return likelihood_func(sub_data, x) - log_prior(x)

        # Run optimization
        res = minimize(
            neg_log_posterior,
            init_params,
            bounds=bounds,
            method="L-BFGS-B",
            options={"maxiter": max_iterations},
        )

        print(f"Completed MAP optimization for agent {aid}")
        return aid, res.x
    except Exception as e:
        print(f"Error processing agent {aid}: {str(e)}")
        return aid, None
