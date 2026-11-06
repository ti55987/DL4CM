"""Noisy RL model of Findling et al. (2019, Nature Neuroscience): Rescorla-Wagner learning whose updates are
corrupted by noise proportional to the prediction error (epsilon), softmax choice with a repetition bias (rep_bias),
and counterfactual learning of the unchosen option, on a two-armed bandit whose payoffs follow random walks.

Simulator adapted from the paper's code, github.com/csmfindling/learning_variability. Task of the paper's
experiment 3: 16 blocks of 56 trials, complete feedback, every choice free; blocks 5 to 8 replay the payoffs of
blocks 1 to 4, and blocks 13 to 16 those of 9 to 12, as in the paper.
"""
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.stats import beta as Beta

N_TRIALS_PER_BLOCK = 56
N_BLOCKS = 16


def generate_noise_rl_parameters(n_agents, default_params={}):
    default_params = {
        "beta": 35,
    }

    alpha_c_dist = np.random.uniform(0.05, 0.8, n_agents)
    alpha_u_dist = np.random.uniform(0.05, 0.8, n_agents)
    epsilon_dist = np.random.uniform(0.01, 1.0, n_agents)
    rep_bias_dist = np.random.uniform(-0.25, 0.5, n_agents)

    return [
        {
            "alpha_c": alpha_c_dist[i],
            "alpha_u": alpha_u_dist[i],
            "beta": default_params["beta"],
            "epsilon": epsilon_dist[i],
            "rep_bias": rep_bias_dist[i],
        }
        for i in range(n_agents)
    ]


def random_walk(T, points_range=[0.1, 0.99], tau=3, omega=1.5):
    r_hat = np.zeros(T)
    r = np.zeros(T)
    for t in range(T):
        if t == 0:
            r_hat[t] = np.random.uniform(*points_range)
        else:
            alpha_hat = 1 + r_hat[t - 1] * np.exp(tau)
            beta_hat = 1 + (1 - r_hat[t - 1]) * np.exp(tau)
            r_hat[t] = Beta(a=alpha_hat, b=beta_hat).rvs()

        alpha = 1 + r_hat[t] * np.exp(omega)
        beta = 1 + (1 - r_hat[t]) * np.exp(omega)
        r[t] = Beta(a=alpha, b=beta).rvs()
    return r


def replay_order(n_blocks, group=4):
    """The payoff walk shown in each block: groups of `group` new walks, each group followed by its replay.
    16 blocks: 0 1 2 3 0 1 2 3 4 5 6 7 4 5 6 7."""
    half = n_blocks // 2
    return np.concatenate(
        [np.tile(np.arange(g, min(g + group, half)), 2) for g in range(0, half, group)]
    )


def task_setup():
    """Payoffs of the two options, shape (2, N_BLOCKS * N_TRIALS_PER_BLOCK): a new pair of random walks for each of
    the first N_BLOCKS // 2 blocks, shown in the order of replay_order."""
    seeds0 = [random_walk(N_TRIALS_PER_BLOCK) for _ in range(N_BLOCKS // 2)]
    seeds1 = [random_walk(N_TRIALS_PER_BLOCK) for _ in range(N_BLOCKS // 2)]
    order = replay_order(N_BLOCKS)
    return np.vstack([np.hstack([seeds0[i] for i in order]), np.hstack([seeds1[i] for i in order])])


def simulate_noisy_rl(rewards, sample):
    """Simulate one agent on the task.

    rewards: payoffs of the two options, shape (2, T), from task_setup.
    sample: [alpha_c, alpha_u, beta, epsilon, rep_bias].

    On the first trial of a block both values reset to 0.5 and the choice is random. On every other trial:
    1. The agent sees both payoffs of the previous trial (complete feedback).
    2. Each value moves toward its payoff, with learning rate alpha_c for the option chosen on the previous trial and
       alpha_u for the other one: mu = (1 - alpha) * Q + alpha * payoff.
    3. Noise is added to each update, with a standard deviation proportional to the size of its prediction error:
       Q = mu + epsilon * |payoff - Q| * N(0, 1).
    4. The choice is a softmax on the two values with inverse temperature beta, plus a bias of rep_bias * beta
       toward repeating the previous choice.

    Returns the values Q (T, 2), the actions (T,), the payoffs seen on each trial (2, T; this trial's payoffs, which
    the agent learns from on the next trial), and whether each action took the option of higher value (T,).
    """
    T = rewards.shape[1]
    noisy_trajectories = np.zeros([T, 2])
    actions_simul = np.zeros([T])
    alpha_c, alpha_u, beta_softmax, epsilon = sample[:4]
    rep_bias = sample[4] * beta_softmax
    observed_rewards = rewards.copy()
    greedy_choices = np.zeros(T)

    for t_idx in range(T):
        if t_idx % N_TRIALS_PER_BLOCK == 0:  # first trial of a block
            noisy_trajectories[t_idx] = 0.5
            actions_simul[t_idx] = 1 if np.random.rand() < 0.5 else 0
            prev_act = actions_simul[t_idx]
            continue

        prev_rew = rewards[:, t_idx - 1]
        if actions_simul[t_idx - 1] == 0:
            mu0 = (1 - alpha_c) * noisy_trajectories[t_idx - 1, 0] + alpha_c * prev_rew[0]
            mu1 = (1 - alpha_u) * noisy_trajectories[t_idx - 1, 1] + alpha_u * prev_rew[1]
        else:
            mu0 = (1 - alpha_u) * noisy_trajectories[t_idx - 1, 0] + alpha_u * prev_rew[0]
            mu1 = (1 - alpha_c) * noisy_trajectories[t_idx - 1, 1] + alpha_c * prev_rew[1]

        noise_level0 = np.abs(prev_rew[0] - noisy_trajectories[t_idx - 1, 0]) * epsilon
        noise_level1 = np.abs(prev_rew[1] - noisy_trajectories[t_idx - 1, 1]) * epsilon
        noisy_trajectories[t_idx, 0] = mu0 + noise_level0 * np.random.normal()
        noisy_trajectories[t_idx, 1] = mu1 + noise_level1 * np.random.normal()

        # probability to choose 1, clipped to avoid numerical issues
        lgt = beta_softmax * (noisy_trajectories[t_idx, 0] - noisy_trajectories[t_idx, 1])
        lgt -= np.sign(prev_act - 0.5) * rep_bias
        proba_1 = 1.0 / (1.0 + np.exp(np.clip(lgt, -30, 30)))

        actions_simul[t_idx] = 1 if np.random.rand() < proba_1 else 0
        prev_act = actions_simul[t_idx]

        if actions_simul[t_idx] == np.argmax(noisy_trajectories[t_idx]):
            greedy_choices[t_idx] = 1

    return noisy_trajectories, actions_simul, observed_rewards, greedy_choices


def simulate_noisy_rl_agent(a, params_dist):
    rewards = task_setup()
    sample = [params_dist[p] for p in ["alpha_c", "alpha_u", "beta", "epsilon", "rep_bias"]]
    trajectories, actions, obs_rewards, greedy_choices = simulate_noisy_rl(rewards, sample)

    return pd.DataFrame(
        {
            "subj_code": a,
            "trial": np.tile(np.arange(N_TRIALS_PER_BLOCK), N_BLOCKS),
            "block": np.repeat(np.arange(N_BLOCKS), N_TRIALS_PER_BLOCK),
            "reward0": rewards[0],
            "reward1": rewards[1],
            "action": actions,
            "greedy_choice": greedy_choices,
            "obs_reward0": obs_rewards[0],
            "obs_reward1": obs_rewards[1],
            "q_value0": trajectories[:, 0],
            "q_value1": trajectories[:, 1],
            "alpha_c": sample[0],
            "alpha_u": sample[1],
            "beta": sample[2],
            "epsilon": sample[3],
            "rep_bias": sample[4],
        }
    )


def simulate_noisy_rl_agents(parameters_dist, max_workers=1):
    """One row per trial for every agent in `parameters_dist` (from generate_noise_rl_parameters).

    max_workers is 1 by default. All random draws, the payoff walks included, come from numpy's global random
    state; on Linux (Colab included) worker processes are forked and start from the same state, so agents simulated
    on different workers can receive identical payoff walks.
    """
    if max_workers > 1 and len(parameters_dist) > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            agent_dfs = list(executor.map(simulate_noisy_rl_agent, range(len(parameters_dist)), parameters_dist))
    else:
        agent_dfs = [simulate_noisy_rl_agent(a, p) for a, p in enumerate(parameters_dist)]

    sim_df = pd.concat(agent_dfs, axis=0, ignore_index=True)
    sim_df = sim_df.sort_values(["subj_code", "block", "trial"]).reset_index(drop=True)

    sim_df["optimal_choice"] = (
        sim_df.action == sim_df[["reward0", "reward1"]].idxmax(axis=1).str[-1].astype(int)
    ).astype(int)
    sim_df["prev_action"] = sim_df.groupby("subj_code")["action"].shift(1)
    sim_df["prev_optimal_choice"] = sim_df.groupby("subj_code")["optimal_choice"].shift(1)
    same_agent = sim_df["subj_code"].eq(sim_df["subj_code"].shift(1))
    sim_df["stay"] = np.where(same_agent, sim_df["prev_action"].eq(sim_df["action"]).astype(int), np.nan).astype(float)
    sim_df["prev_reward_val"] = np.where(
        same_agent,
        np.where(sim_df["prev_action"] == 0, sim_df["reward0"].shift(1), sim_df["reward1"].shift(1)),
        np.nan,
    )
    sim_df["reward_diff"] = sim_df.reward1 - sim_df.reward0
    sim_df["prev_reward_diff"] = sim_df.groupby("subj_code")["reward_diff"].shift(1)
    sim_df["observed_reward_diff"] = sim_df.obs_reward1 - sim_df.obs_reward0
    sim_df["prev_observed_reward_diff"] = sim_df.groupby("subj_code")["observed_reward_diff"].shift(1)
    sim_df["q_diff"] = sim_df.q_value1 - sim_df.q_value0
    sim_df["prev_q_diff"] = sim_df.groupby("subj_code")["q_diff"].shift(1)
    sim_df["true_best"] = (sim_df["reward1"] > sim_df["reward0"]).astype(int)
    sim_df["is_switch"] = (
        sim_df.groupby(["subj_code", "block"])["true_best"].diff().fillna(0).abs().gt(0).astype(int)
    )
    sim_df["trials_since_switch"] = sim_df.groupby(["subj_code", "block"])["is_switch"].cumsum()
    sim_df["trial_post_switch"] = sim_df.groupby(["subj_code", "block", "trials_since_switch"]).cumcount()

    return sim_df.astype(float)
