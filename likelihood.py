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


def rl3_sa_neg_log_likelihood(data, param_dict):
    param_dict = {
        "beta": param_dict["beta"],
        "alpha0": param_dict["alpha0"],
        "alpha1": param_dict["alpha1"],
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


def get_prl4_rpe(stimuli, actions, rewards, param_dict, num_actions=3, num_stimuli=6):
    parameters = {
        "alpha": param_dict["alpha"],
        "beta": param_dict.get("beta", 2.5),
        "stickiness": param_dict["stickiness"],
        "phi": param_dict.get("phi", 0),
        "bias": param_dict.get("bias", 0),
    }
    beta = parameters["beta"] * 10  # why do it here?
    neg_alpha = param_dict.get("neg_alpha", parameters["alpha"])

    q_values = {
        s: {a: 1 / num_actions for a in range(num_actions)} for s in range(num_stimuli)
    }  # equal value first
    lr_list = [neg_alpha, parameters["alpha"]]
    rpe_history, chosen_q_values = [], []
    for s, a, r in zip(stimuli, actions, rewards):
        # Forgetting - fix to case with different Q/W
        for st, action_to_prob in q_values.items():
            for ac in action_to_prob.keys():
                # same thing as WM = WM + forget (1/n - WM)
                q_values[st][ac] = (1.0 - parameters["phi"]) * q_values[st][
                    ac
                ] + parameters["phi"] * 1 / num_actions

        rpe = r - q_values[s][a]
        alpha = lr_list[r]
        # print("s, a, r", s, a, r, q_values[s][a])
        chosen_q_values.append(q_values[s][a])
        rpe_history.append(rpe)

        # Q updates
        q_values[s][a] = q_values[s][a] + alpha * rpe
        # action that's not selected (for counterfactual learning)
        for x in list(np.arange(num_actions)):
            if x == a:
                continue

            rpe_unchosen = (1 - r) - q_values[s][x]  # RPE for the unselected action
            q_values[s][x] += alpha * rpe_unchosen

    return rpe_history, chosen_q_values


# 2PRL-SA likelihood
def sa_neg_log_likelihood_v2(data, param_dict):
    from rl_models import PRL

    alpha = param_dict["alpha"] if "alpha" in param_dict else 1
    alpha0 = param_dict["alpha0"] if "alpha0" in param_dict else alpha
    alpha1 = param_dict["alpha1"] if "alpha1" in param_dict else alpha
    alpha_cond = {
        0: alpha0,
        1: alpha1,
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
    for _, b_data in data.groupby("block_no"):
        condition = b_data.condition.iloc[0]

        neg_alpha = (
            param_dict["neg_alpha"]
            if "neg_alpha" in param_dict
            else alpha_cond[condition]
        )
        agent.init_model(
            alpha=alpha_cond[condition],
            neg_alpha=neg_alpha,
            stimuli=np.arange(num_stimuli),
            actions=np.arange(num_actions),
            mapping={},
        )
        prev_a = -1
        for _, row in b_data.iterrows():
            ac = int(row.actions)
            st = int(row.stimuli)
            r = int(row.rewards)
            llh += np.log(agent.get_policy(st, prev_a)[ac])
            prev_a = ac
            agent.update_values(st, ac, r)

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


def optimize_MLE(
    data, param_bounds_dict, likelihood_func, n_tries=8, max_iterations=1000
):
    """
    Optimizes model parameters using a dictionary for parameter bounds and values.

    Args:
        data: Dataset to fit the model to
        param_bounds_dict: Dictionary of parameter names to (lower, upper) bound tuples
        likelihood_func: Function that calculates negative log likelihood
        adapative_init_reward: Whether to use adaptive initial reward
        n_tries: Number of random initializations to try

    Returns:
        best_res: Best optimization result
        best_history: History of likelihood values for best run
        all_likelihoods: Final likelihood values for all runs
    """
    best_res, best_history, all_likelihoods = None, [], []
    best_nll = np.inf

    # Get parameter names and bounds
    param_names = sorted(list(param_bounds_dict.keys()))
    bounds = [param_bounds_dict[param] for param in param_names]

    # Create a wrapper function that unpacks dictionary to list for the likelihood function
    def func(params_list, *args):
        # Convert params list back to dictionary for tracking/debugging
        params_dict = {name: value for name, value in zip(param_names, params_list)}
        return likelihood_func(data, params_dict)

    # Try multiple random initializations
    for _ in range(n_tries):
        # func = lambda x, *args: likelihood_func(data, x)

        history = []

        def callback(x):
            history.append(func(x))

        init_params = [random.uniform(l, h) for l, h in bounds]
        res = minimize(
            func,
            init_params,
            bounds=bounds,
            method="L-BFGS-B",
            options={"maxiter": max_iterations},
            callback=callback,
        )
        # print(res)
        all_likelihoods.append(res.fun)
        if res.fun < best_nll:
            best_nll = res.fun
            best_res = res
            best_history = history

    print("best_res", best_res.success, best_res.fun)
    return best_res, best_history, all_likelihoods


def process_agent_v2(
    data_model,
    fit_model,
    agent_data,
    param_bounds_dict,
    likelihood_func,
    n_tries=8,
    max_iterations=1000,
):
    """
    Process a single agent using a specific model setting.

    Args:
        data_model: The model name that generated this data
        agent_data: The agent's data
        param_bounds_dict: Dictionary containing parameter bounds for the model
        model_name: The name of the model to use
    Returns:
        Dictionary containing the recovered parameters and fit metrics
    """
    results = []
    agent_id = agent_data.agentid.unique()[0]
    print("max_iterations", max_iterations, "n_tries", n_tries)
    try:
        print(f"Fitting agent {agent_id}")

        # Check if we have data for this agent
        if len(agent_data) == 0:
            print(f"No data found for agent {agent_id}")
            return []

        # Optimize parameters
        best_res, best_history, all_likelihoods = optimize_MLE(
            agent_data,
            param_bounds_dict,
            likelihood_func,
            max_iterations=max_iterations,
            n_tries=n_tries,
        )

        # Check optimization status
        llh = best_res.fun
        if not best_res.success:
            print(f"Warning: Optimization failed for {agent_id}")
            print(f"Message: {best_res.message}, llh: {llh}")

        # Calculate fit metrics
        n_data_points = len(agent_data)
        n_params = len(get_free_parameters(param_bounds_dict))
        # Calculate AIC and BIC
        aic = calculate_aic(n_params, -llh)  # 2 * n_params + 2 * best_res.fun
        bic = calculate_bic(n_data_points, n_params, -llh)
        # Prepare result dictionary
        result = {
            "id": agent_id,
            "data_model": data_model,  # The model that generated this data
            "fit_model": fit_model,
            "llh": llh,
            "aic": aic,
            "bic": bic,
            "history": best_history,
            "all_likelihoods": all_likelihoods,
            "params": best_res.x,
            "param_names": sorted(list(param_bounds_dict.keys())),
        }

        results.append(result)

    except Exception as e:
        print(f"Error processing agent {agent_id}:{str(e)}")

    return results

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
