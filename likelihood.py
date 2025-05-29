import numpy as np
import random

import scipy
from scipy.optimize import minimize


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


def rl2_sa_neg_log_likelihood(data, parameters):
    alpha0, beta = parameters
    param_dict = {"alpha": alpha0, "beta": beta, "stickiness": 0, "phi": 0, "bias": 1}
    return sa_neg_log_likelihood(data, param_dict)


def wm3_sa_neg_log_likelihood(data, parameters):
    beta, sticky, phi = parameters
    param_dict = {"alpha": 1, "beta": beta, "stickiness": sticky, "phi": phi, "bias": 1}
    return sa_neg_log_likelihood(data, param_dict)


def wmb_sa_neg_log_likelihood(data, parameters):
    beta, sticky, phi, bias = parameters
    param_dict = {
        "alpha": 1,
        "beta": beta,
        "stickiness": sticky,
        "phi": phi,
        "bias": bias,
    }
    return sa_neg_log_likelihood(data, param_dict)


def wmn_sa_neg_log_likelihood(data, parameters):
    beta, sticky, phi, eps = parameters
    param_dict = {
        "alpha": 1,
        "beta": beta,
        "stickiness": sticky,
        "phi": phi,
        "bias": 1,
        "eps": eps,
    }
    return sa_neg_log_likelihood(data, param_dict)


def rl3_sa_neg_log_likelihood(data, parameters):
    alpha0, beta, sticky = parameters
    param_dict = {
        "alpha": alpha0,
        "beta": beta,
        "stickiness": sticky,
        "phi": 0,
        "bias": 1,
    }
    return sa_neg_log_likelihood(data, param_dict)


def rl4_sa_neg_log_likelihood(data, parameters):
    alpha0, beta, sticky, phi = parameters
    param_dict = {
        "alpha": alpha0,
        "beta": beta,
        "stickiness": sticky,
        "phi": phi,
        "bias": 1,
    }
    return sa_neg_log_likelihood(data, param_dict)

# 2PRL-SA likelihood
def sa_neg_log_likelihood_v2(data, param_dict):
    from rl_models import PRL
    
    alpha_cond = {
        0: param_dict["alpha"],
        1: param_dict["alpha"],
    }

    num_actions = len(data.actions.unique())
    num_stimuli = len(data.stimuli.unique())
    agent = PRL(
        beta=param_dict["beta"] * BETA_MULTIPLIER,
        pval=1,
        id=a,
        phi=param_dict["phi"],
        stickiness=param_dict["stickiness"],
        bias=param_dict["bias"],
        eps=param_dict["eps"],
    )
    llh = 0
    for b in data.block_no.unique():
        block_data = data[data.block_no == b]
        condition = block_data.condition.iloc[0]
        agent.init_model(
            alpha=alpha_cond[condition],
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

# 2PRL-SA likelihood
def sa_neg_log_likelihood(data, param_dict):
    alpha0, beta, sticky, phi, bias = (
        param_dict["alpha"],
        param_dict["beta"],
        param_dict["stickiness"],
        param_dict["phi"],
        param_dict["bias"],
    )
    alpha1 = alpha0
    beta = beta * BETA_MULTIPLIER
    alpha_cond = {
        0: [alpha0 * bias, alpha0],
        1: [alpha1 * bias, alpha1],
    }

    num_actions = len(data.actions.unique())
    llh = 0
    for b in data.block_no.unique():
        block_data = data[data.block_no == b]
        condition = block_data.condition.iloc[0]
        alphas = alpha_cond[condition]

        num_stimuli = len(block_data.stimuli.unique())
        init_value = 1.0 / num_actions
        q_values = {
            i: np.array([init_value] * num_actions) for i in range(num_stimuli)
        }  # equal value first
        prev_a = -1
        for s, a, r in zip(block_data.stimuli, block_data.actions, block_data.rewards):
            Q = q_values.copy()
            if prev_a != -1:
                Q[s][prev_a] = Q[s][prev_a] + sticky

            llh += np.log(scipy.special.softmax(beta * Q[s])[a])
            prev_a = a
            # llh += np.log(scipy.special.softmax(beta * q_values[s])[a])

            # Forgetting - fix to case with different Q/W
            for st, action_to_prob in q_values.items():
                for i in range(len(action_to_prob)):
                    # same thing as WM = WM + forget (1/n - WM)
                    q_values[st][i] = (1.0 - phi) * q_values[st][i] + phi * init_value

            rpe = r - q_values[s][a]
            alpha = alphas[r]
            q_values[s][a] += alpha * rpe  # update q value
            for x in list(np.arange(num_actions)):
                if x == a:
                    continue
                # RPE for the unselected action
                rpe_unchosen = (1 - r) - q_values[s][x]
                q_values[s][x] += alpha * rpe_unchosen

    return -llh


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
            return self.log_prob
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


# Function to process a single agent ID
def process_agent(
    aid,
    data,
    metadata,
    bound_name="bounds",
    max_iterations=30,
):
    """Process a single agent ID and return the optimization results."""
    likelihood_func = metadata["likelihood_func"]
    try:
        print(f"Starting optimization for agent {aid}...")
        sub_data = data[data.agentid == aid]
        init_params = [random.uniform(l, h) for l, h in metadata[bound_name]]

        # Define the function for this agent
        func = lambda x, *args: likelihood_func(sub_data, x)

        # Run optimization
        res = minimize(
            func,
            init_params,
            bounds=metadata[bound_name],
            method="L-BFGS-B",
            options={"maxiter": max_iterations},
        )

        print(f"Completed optimization for agent {aid}")
        return aid, res.x
    except Exception as e:
        print(f"Error processing agent {aid}: {str(e)}")
        return aid, None


def process_agent_map(
    aid,
    data,
    metadata,
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
    likelihood_func = metadata["likelihood_func"]
    try:
        print(f"Starting MAP optimization for agent {aid}...")
        sub_data = data[data.agentid == aid]
        init_params = [random.uniform(l, h) for l, h in metadata["bounds"]]

        # Define the uniform prior log probability function
        def log_prior(params):
            log_prob = 0.0
            for i, prior_func in enumerate(metadata["prior_func"]):
                log_prob += prior_func(params[i])
            return log_prob

        # Define the function for this agent (negative log posterior)
        def neg_log_posterior(x, *args):
            # Negative log posterior = negative log likelihood - log prior
            return likelihood_func(sub_data, x) - log_prior(x)

        # Run optimization
        res = minimize(
            neg_log_posterior,
            init_params,
            bounds=metadata["bounds"],
            method="L-BFGS-B",
            options={"maxiter": max_iterations},
        )

        print(f"Completed MAP optimization for agent {aid}")
        return aid, res.x
    except Exception as e:
        print(f"Error processing agent {aid}: {str(e)}")
        return aid, None
