### -------------------------------------------------------------------------
# Noisy RL utils adapted from https://github.com/gaiamolinaro/noisyRL/blob/main/simulate_agents.ipynb
### -------------------------------------------------------------------------
import random
from functools import partial
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from scipy.stats import beta as Beta
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import MinMaxScaler


def get_filenames(
    out_dir,
    model_name,
    experiment,
    n_agents_train,
    n_agents_test,
    apply_rep_bias,
    apply_weber,
    apply_observational_noise,
    apply_counterfactual_learn,
    fmt="csv",
    train_test=True,
):
    fname_train = f"{out_dir}sim_data_{model_name}_exp{experiment}_nAgents{n_agents_train}_repBias{int(apply_rep_bias)}_weber{int(apply_weber)}_obsNoise{int(apply_observational_noise)}_cfLearn{int(apply_counterfactual_learn)}.{fmt}"
    fname_test = f"{out_dir}sim_data_{model_name}_exp{experiment}_nAgents{n_agents_test}_repBias{int(apply_rep_bias)}_weber{int(apply_weber)}_obsNoise{int(apply_observational_noise)}_cfLearn{int(apply_counterfactual_learn)}.{fmt}"
    if train_test:
        return fname_train, fname_test
    else:
        return fname_train


def set_plot_params():
    """
    Set parameters for plotting
    """
    font_size = 14
    k_color = "k"
    bg_color = "w"
    plt.rcParams["errorbar.capsize"] = 5
    legend_font_size = font_size - 2
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = font_size

    plt.rcParams["lines.color"] = k_color
    plt.rcParams["lines.markeredgecolor"] = k_color
    plt.rcParams["lines.markerfacecolor"] = k_color
    plt.rcParams["axes.edgecolor"] = k_color
    plt.rcParams["text.color"] = k_color
    plt.rcParams["axes.labelcolor"] = k_color
    plt.rcParams["xtick.color"] = k_color
    plt.rcParams["ytick.color"] = k_color
    plt.rcParams["lines.color"] = k_color
    plt.rcParams["axes.facecolor"] = bg_color
    plt.rcParams["figure.facecolor"] = bg_color

    plt.rcParams["legend.frameon"] = False
    plt.rcParams["legend.labelspacing"] = 0
    plt.rcParams["legend.columnspacing"] = 0.7
    plt.rcParams["legend.handletextpad"] = 0.1

    plt.rcParams["lines.linewidth"] = 2
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.default"] = "regular"
    plt.rcParams["mathtext.rm"] = "Arial"
    plt.rcParams["mathtext.it"] = "Arial:italic"
    plt.rcParams["mathtext.bf"] = "Arial:bold"
    return


def get_param_sets(alpha_cs, alpha_us, betas, epsilons, rep_biases, n_iters=0):
    # Generate a meshgrid of all parameter combinations
    # indexing='ij' ensures the order follows the input lists (Matrix indexing)
    grid = np.meshgrid(alpha_cs, alpha_us, betas, epsilons, rep_biases, indexing="ij")
    # stack and reshape to get the (N, 5) shape
    param_sets = np.stack(grid, axis=-1).reshape(-1, 5)
    if n_iters > 0:
        param_sets = np.tile(param_sets, n_iters)
    return param_sets


def plot_p_event_vs_var(sim_df, param1, param2, xvar="prev_reward_val", yvar="stay"):
    xvar_name_mapping = {
        "prev_reward_val": "Previous reward value",
        "prev_reward_diff": "Previous reward difference (R1-R0)",
        "prev_observed_reward_diff": "Previous observed reward difference (R1-R0)",
        "prev_q_diff": "Previous Q-value difference (Q1-Q0)",
    }
    yvar_name_mapping = {
        "stay": "P(stay)",
        "action": "P(action=1)",
    }

    # 1. Setup Parameters
    param1s = sorted(sim_df[param1].unique())
    param2s = sorted(sim_df[param2].unique())
    n_param1s = len(param1s)

    # Create a color palette based on param2 values
    palette = sns.color_palette("Spectral", n_colors=len(param2s))

    # 2. Initialize the Grid
    fig, axes = plt.subplots(1, n_param1s, figsize=(5 * n_param1s, 5), sharey=True)

    # Handle cases with only one param1 column
    if n_param1s == 1:
        axes = [axes]

    # 3. Iterate through param1 (columns) and param2 (colors)
    for i, a_val in enumerate(param1s):
        ax = axes[i]
        param1_subset = sim_df[sim_df[param1] == a_val].dropna(subset=["stay", xvar])

        for j, b_val in enumerate(param2s):
            param2_subset = param1_subset[param1_subset[param2] == b_val]

            if not param2_subset.empty:
                # We use a label here so the legend can find the "artist"
                sns.regplot(
                    data=param2_subset,
                    x=xvar,
                    y=yvar,
                    logistic=True,
                    scatter=False,
                    ci=95,
                    line_kws={"color": palette[j], "linewidth": 2.5},
                    ax=ax,
                    label=b_val,
                )

        ax.set_title(rf"$\{param1}$ = {a_val}")
        ax.set_xlabel(xvar_name_mapping.get(xvar, xvar))
        ax.set_ylim(-0.02, 1.02)

        if i == 0:
            ax.set_ylabel(yvar_name_mapping.get(yvar, yvar))
            ax.legend(title=rf"$\{param2}$", loc="lower right")
        else:
            ax.set_ylabel("")

    plt.tight_layout()
    return fig, axes


def plot_switch_adaptation(sim_df, param1="alpha_c", param2="beta", max_trials=15):

    # Filter for the window immediately following a switch (excluding the very first trials)
    # We also exclude trials_since_switch == 0 (the period before the first switch)
    adaptation_df = sim_df[
        (sim_df["trial_post_switch"] < max_trials) & (sim_df["trials_since_switch"] > 0)
    ]

    col_vals = adaptation_df[param1].unique()
    hue_vals = adaptation_df[param2].unique()

    n_cols = len(col_vals)
    palette = sns.color_palette("Spectral", n_colors=len(hue_vals))

    fig, axes = plt.subplots(1, n_cols, figsize=(3 * n_cols, 3), sharey=True)
    if n_cols == 1:
        axes = [axes]

    for i, col_val in enumerate(col_vals):
        ax = axes[i]
        subset = adaptation_df[adaptation_df[param1] == col_val]
        sns.lineplot(
            data=subset,
            x="trial_post_switch",
            y="optimal_choice",
            hue=param2,
            palette=palette,
            ax=ax,
            legend=i == 0,
            linewidth=2.5,
        )

        ax.set_title(f"{param1}= {col_val}")
        ax.set_xlabel("Trials since reward crossover")
        ax.set_ylim(0, 1.0)
        ax.axhline(0.5, ls="--", color="gray", alpha=0.5)

        if i == 0:
            ax.set_ylabel("P(choose new optimal)")
            ax.legend(title=f"{param2}", loc="lower left")

    plt.tight_layout()
    return fig, axes


def get_mi_bits(df):
    # Mask for transitions where both current and previous trials were free choices
    # and were within the same block

    mask = (
        (df["free_choice"] == 1)
        & (df["free_choice"].shift() == 1)
        & (df.block.shift(1).eq(df.block))
    )

    a_t = df["action"][mask]
    a_t_minus_1 = df["action"].shift()[mask]

    if len(a_t) < 2:
        return 0

    # Calculate in nats then convert to bits
    return mutual_info_score(a_t_minus_1, a_t) / np.log(2)


def plot_mi_vs_params(sim_df, param, fig=None, ax=None):
    scaler = MinMaxScaler()
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    plot_df = (
        sim_df.groupby(["subj_code", param])
        .apply(get_mi_bits)
        .reset_index()
        .rename(columns={0: "mutual_info"})
    )
    plot_df[f"{param}_rescaled"] = scaler.fit_transform(plot_df[[param]])
    sns.regplot(
        data=plot_df,
        x=f"{param}_rescaled",
        y="mutual_info",
        scatter_kws={"s": 10},
        # label=rf"$\{param}$" if param != "rep_bias" else r"$\xi$",
        ax=ax,
    )
    ax.legend()
    ax.set_ylabel("Mutual information")
    ax.set_title(rf"$\{param}$" if param != "rep_bias" else r"$\xi$")
    return fig, ax, plot_df


# Modified from twoq_value_2alpha.simulation.simulation
def simulate_noisy_rl(
    rewards,
    complete,
    idx_blocks,
    choices,
    forced_actions,
    sample,
    apply_rep_bias,
    apply_weber,
    apply_observational_noise,
    apply_counterfactual_learn,
):
    T = len(choices)
    noisy_trajectories = np.zeros([T, 2])
    actions_simul = np.zeros([T])
    prev_act = -1
    alpha_c = sample[0]
    alpha_u = sample[1] if apply_counterfactual_learn else 0
    beta_softmax = sample[2]
    epsilon = sample[3]
    rep_bias = sample[4] * beta_softmax if apply_rep_bias else 0
    prev_rew = np.zeros(2)
    observed_rewards = np.zeros(rewards.shape) - np.inf
    greedy_choices = np.zeros(T)

    for t_idx in range(T):

        if t_idx > 0:
            if complete[t_idx - 1]:
                prev_rew = rewards[:, t_idx - 1]
            else:
                prev_rew[int(actions_simul[t_idx - 1])] = rewards[
                    int(actions_simul[t_idx - 1]), t_idx - 1
                ]
                prev_rew[1 - int(actions_simul[t_idx - 1])] = 0.5

            observed_rewards[:, t_idx - 1] = prev_rew

        if idx_blocks[t_idx] == 0:  # normal trial

            if actions_simul[t_idx - 1] == 0:
                mu0 = (1 - alpha_c) * noisy_trajectories[
                    t_idx - 1, 0
                ] + alpha_c * prev_rew[0]
                mu1 = (1 - alpha_u) * noisy_trajectories[
                    t_idx - 1, 1
                ] + alpha_u * prev_rew[1]
            else:
                mu0 = (1 - alpha_u) * noisy_trajectories[
                    t_idx - 1, 0
                ] + alpha_u * prev_rew[0]
                mu1 = (1 - alpha_c) * noisy_trajectories[
                    t_idx - 1, 1
                ] + alpha_c * prev_rew[1]

            if apply_weber == 1:
                noise_level0 = (
                    np.abs(prev_rew[0] - noisy_trajectories[t_idx - 1, 0]) * epsilon
                )
                noise_level1 = (
                    np.abs(prev_rew[1] - noisy_trajectories[t_idx - 1, 1]) * epsilon
                )
            else:
                noise_level0 = epsilon
                noise_level1 = epsilon

            if apply_observational_noise == 0 and choices[t_idx - 1] == 0:
                noise_level0 = 0
                noise_level1 = 0

            noisy_trajectories[t_idx, 0] = mu0 + noise_level0 * np.random.normal()
            noisy_trajectories[t_idx, 1] = mu1 + noise_level1 * np.random.normal()

            if choices[t_idx] == 1:
                # probability to choose 1, clipped to avoid numerical issues
                if (apply_rep_bias == 0) or (prev_act == -1):
                    lgt = beta_softmax * (
                        noisy_trajectories[t_idx, 0] - noisy_trajectories[t_idx, 1]
                    )
                    proba_1 = 1.0 / (1.0 + np.exp(np.clip(lgt, -30, 30)))
                else:
                    lgt = (
                        beta_softmax
                        * (noisy_trajectories[t_idx, 0] - noisy_trajectories[t_idx, 1])
                        - np.sign(prev_act - 0.5) * rep_bias
                    )
                    proba_1 = 1.0 / (1.0 + np.exp(np.clip(lgt, -30, 30)))

                # simulate action
                if np.random.rand() < proba_1:
                    actions_simul[t_idx] = 1
                else:
                    actions_simul[t_idx] = 0

                # previous action
                prev_act = actions_simul[t_idx]
            else:
                actions_simul[t_idx] = forced_actions[t_idx]

            if actions_simul[t_idx] == np.argmax(noisy_trajectories[t_idx]):
                greedy_choices[t_idx] = 1

        else:  # block reset
            noisy_trajectories[t_idx] = 0.5
            prev_act = -1

            if choices[t_idx] == 1:
                if np.random.rand() < 0.5:
                    actions_simul[t_idx] = 1
                else:
                    actions_simul[t_idx] = 0

                # previous action
                prev_act = actions_simul[t_idx]
            else:
                actions_simul[t_idx] = forced_actions[t_idx]

    if t_idx > 0:
        if complete[t_idx]:
            prev_rew = rewards[:, t_idx]
        else:
            prev_rew[int(actions_simul[t_idx])] = rewards[
                int(actions_simul[t_idx]), t_idx
            ]
            prev_rew[1 - int(actions_simul[t_idx])] = 0.5

        observed_rewards[:, t_idx] = prev_rew

    return noisy_trajectories, actions_simul, observed_rewards, greedy_choices


# simulate standard noise-free RL model
def simulate_noiseless_rl(
    rewards,
    complete,
    idx_blocks,
    choices,
    forced_actions,
    sample,
    apply_rep_bias,
    apply_weber_decision_noise,
    apply_observational_noise,
    apply_counterfactual_learn,
):
    assert (
        apply_weber_decision_noise == 0
    ), "Simulation not implemented for apply_weber_decision_noise = 1"
    assert (
        apply_observational_noise == 0
    ), "Simulation doesn't support apply_observational_noise = 1"

    T = len(choices)
    noiseless_trajectories = np.zeros([T, 2])
    actions_simul = np.zeros([T])
    prev_act = -1
    alpha_c = sample[0]
    alpha_u = sample[1] if apply_counterfactual_learn else 0
    beta_softmax = sample[2]
    epsilon = (
        0  # Not used in this model, but included for consistency in parameter sets
    )
    rep_bias = sample[4] * beta_softmax if apply_rep_bias else 0
    prev_rew = np.zeros(2)
    observed_rewards = np.zeros(rewards.shape) - np.inf
    greedy_choices = np.zeros(T)

    for t_idx in range(T):

        if t_idx > 0:
            # Change: complete is now indexed by time
            if complete[t_idx - 1]:
                prev_rew = rewards[:, t_idx - 1]
            else:
                prev_rew[int(actions_simul[t_idx - 1])] = rewards[
                    int(actions_simul[t_idx - 1]), t_idx - 1
                ]
                prev_rew[1 - int(actions_simul[t_idx - 1])] = 0.5

            observed_rewards[:, t_idx - 1] = prev_rew

        if idx_blocks[t_idx] == 0:  # normal trial
            if actions_simul[t_idx - 1] == 0:
                noiseless_trajectories[t_idx, 0] = (
                    1 - alpha_c
                ) * noiseless_trajectories[t_idx - 1, 0] + alpha_c * prev_rew[0]
                noiseless_trajectories[t_idx, 1] = (
                    1 - alpha_u
                ) * noiseless_trajectories[t_idx - 1, 1] + alpha_u * prev_rew[1]
            else:
                noiseless_trajectories[t_idx, 0] = (
                    1 - alpha_u
                ) * noiseless_trajectories[t_idx - 1, 0] + alpha_u * prev_rew[0]
                noiseless_trajectories[t_idx, 1] = (
                    1 - alpha_c
                ) * noiseless_trajectories[t_idx - 1, 1] + alpha_c * prev_rew[1]
        else:  # block reset
            prev_act = -1
            noiseless_trajectories[t_idx] = 0.5

        if choices[t_idx] == 1:
            # Change: calculate lgt with clipping to avoid numerical issues
            if (apply_rep_bias == 0) or (prev_act == -1):
                lgt = beta_softmax * (
                    noiseless_trajectories[t_idx, 0] - noiseless_trajectories[t_idx, 1]
                )
                proba_1 = 1.0 / (1.0 + np.exp(np.clip(lgt, -30, 30)))
            else:
                lgt = (
                    beta_softmax
                    * (
                        noiseless_trajectories[t_idx, 0]
                        - noiseless_trajectories[t_idx, 1]
                    )
                    - np.sign(prev_act - 0.5) * rep_bias
                )
                proba_1 = 1.0 / (1.0 + np.exp(np.clip(lgt, -30, 30)))

            if np.random.rand() < proba_1:
                actions_simul[t_idx] = 1
            else:
                actions_simul[t_idx] = 0

            prev_act = actions_simul[t_idx]
        else:
            actions_simul[t_idx] = forced_actions[t_idx]

        # Change: Add greedy choice tracking
        if actions_simul[t_idx] == np.argmax(noiseless_trajectories[t_idx]):
            greedy_choices[t_idx] = 1

    if t_idx > 0:
        # Change: complete is now indexed by time
        if complete[t_idx]:
            prev_rew = rewards[:, t_idx]
        else:
            prev_rew[int(actions_simul[t_idx])] = rewards[
                int(actions_simul[t_idx]), t_idx
            ]
            prev_rew[1 - int(actions_simul[t_idx])] = 0.5

        observed_rewards[:, t_idx] = prev_rew

    return noiseless_trajectories, actions_simul, observed_rewards, greedy_choices


def nllh_noiseless_rl(
    sample,
    data,
    apply_rep_bias,
    apply_weber_decision_noise,
    apply_observational_noise,
    apply_counterfactual_learn,
    beta_multiplier,
):
    assert (
        apply_weber_decision_noise == 0
    ), "Likelihood not implemented for apply_weber_decision_noise = 1"
    assert (
        apply_observational_noise == 0
    ), "Likelihood doesn't support apply_observational_noise = 1"

    rewards = data[["reward0", "reward1"]].to_numpy(dtype=float).T
    actions = data["action"].to_numpy(dtype=int)
    actions_free = data["free_choice"].to_numpy(dtype=int)
    complete = data["feedback_type"].to_numpy(dtype=int)
    idx_blocks = np.r_[
        1,
        data["block"].to_numpy(dtype=int)[1:] != data["block"].to_numpy(dtype=int)[:-1],
    ].astype(int)
    forced_actions = np.where(actions_free == 0, actions, 0)

    T = len(actions)
    noiseless_trajectories = np.zeros([T, 2])
    prev_act = -1
    alpha_c = sample[0]
    alpha_u = sample[1] if apply_counterfactual_learn else 0
    beta_softmax = sample[2] * beta_multiplier
    rep_bias = sample[4] * beta_softmax if apply_rep_bias else 0
    prev_rew = np.zeros(2)
    nllh = 0

    for t_idx in range(T):

        if t_idx > 0:
            if complete[t_idx - 1]:
                prev_rew = rewards[:, t_idx - 1].copy()
            else:
                prev_rew[int(actions[t_idx - 1])] = rewards[
                    int(actions[t_idx - 1]), t_idx - 1
                ]
                prev_rew[1 - int(actions[t_idx - 1])] = 0.5

        if idx_blocks[t_idx] == 0:
            if actions[t_idx - 1] == 0:
                noiseless_trajectories[t_idx, 0] = (
                    1 - alpha_c
                ) * noiseless_trajectories[t_idx - 1, 0] + alpha_c * prev_rew[0]
                noiseless_trajectories[t_idx, 1] = (
                    1 - alpha_u
                ) * noiseless_trajectories[t_idx - 1, 1] + alpha_u * prev_rew[1]
            else:
                noiseless_trajectories[t_idx, 0] = (
                    1 - alpha_u
                ) * noiseless_trajectories[t_idx - 1, 0] + alpha_u * prev_rew[0]
                noiseless_trajectories[t_idx, 1] = (
                    1 - alpha_c
                ) * noiseless_trajectories[t_idx - 1, 1] + alpha_c * prev_rew[1]
        else:
            prev_act = -1
            noiseless_trajectories[t_idx] = 0.5

        if actions_free[t_idx] == 1:
            if (apply_rep_bias == 0) or (prev_act == -1):
                lgt = beta_softmax * (
                    noiseless_trajectories[t_idx, 0] - noiseless_trajectories[t_idx, 1]
                )
            else:
                lgt = (
                    beta_softmax
                    * (
                        noiseless_trajectories[t_idx, 0]
                        - noiseless_trajectories[t_idx, 1]
                    )
                    - np.sign(prev_act - 0.5) * rep_bias
                )

            proba_1 = 1.0 / (1.0 + np.exp(np.clip(lgt, -30, 30)))
            proba_1 = np.clip(proba_1, 1e-12, 1 - 1e-12)
            action = int(actions[t_idx])
            nllh -= np.log(proba_1 if action == 1 else 1 - proba_1)
            prev_act = action
        else:
            actions[t_idx] = forced_actions[t_idx]

    return nllh


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


def task_setup(experiment, n_blocks, n_trials_per_block):
    T = n_blocks * n_trials_per_block
    # rewards: 2 rows (options), T columns
    # New random walks start at each block, but the same random walks are repeated in the second half of the experiment
    rewards = np.zeros((2, T))

    if experiment == 2:
        # The first half of the experiment requires n_blocks // 2 blocks.
        n_half = n_blocks // 2

        # counter-balance omegas for the first half: half get 1, half get 2
        # (make sure n_blocks is a multiple of 4, so n_half is even)
        omegas = np.random.permutation(np.repeat([1.0, 2.0], n_half // 2))

        # generate the walks for the first half using the n_half omegas, then tile by 2
        rewards[0] = np.tile(
            np.hstack(
                [
                    random_walk(n_trials_per_block, omega=omegas[i])
                    for i in range(n_half)
                ]
            ),
            2,
        )
        rewards[1] = np.tile(
            np.hstack(
                [
                    random_walk(n_trials_per_block, omega=omegas[i])
                    for i in range(n_half)
                ]
            ),
            2,
        )

    else:
        # experiments 0, 1 & 3: default omega (1.5) is used inside random_walk()
        rewards[0] = np.tile(
            np.hstack([random_walk(n_trials_per_block) for _ in range(n_blocks // 2)]),
            2,
        )
        rewards[1] = np.tile(
            np.hstack([random_walk(n_trials_per_block) for _ in range(n_blocks // 2)]),
            2,
        )

    # complete: 1 for full feedback, 0 for partial (RL standard)
    if experiment in [1, 2]:
        complete = []
        for i in range(n_blocks // 2):
            if np.random.rand() < 0.5:
                complete.append(np.ones(n_trials_per_block))
                complete.append(np.zeros(n_trials_per_block))
            else:
                complete.append(np.zeros(n_trials_per_block))
                complete.append(np.ones(n_trials_per_block))
        complete = np.hstack(complete)
    elif experiment == 0:
        # always partial feedback in experiment 0
        complete = np.zeros(T)
    else:
        # always complete in experiment 3
        complete = np.ones(T)

    # idx_blocks: 0 means normal learning, 1 resets the model
    idx_blocks = np.tile([1] + [0] * (n_trials_per_block - 1), n_blocks)

    # choices: 1 if the agent chooses, 0 if it's a forced action
    if experiment == 1:
        choices = np.ones(T)
        # 25% of trials are cued (forced actions) in experiment 1
        p_forced = 0.25
        n_forced_per_block = int(p_forced * n_trials_per_block)
        n_free_per_block = int((1 - p_forced) * n_trials_per_block)
        choices = np.hstack(
            [
                np.random.permutation([0] * n_forced_per_block + [1] * n_free_per_block)
                for _ in range(n_blocks)
            ]
        )
        idx_cued = np.where(choices == 0)[0]
    else:
        # always free choice in experiments 0, 2 and 3
        choices = np.ones(T)

    # forced_actions: only used if choices[t] == 0
    if experiment == 1:
        forced_actions = np.zeros(T)
        forced_actions[idx_cued] = np.hstack(
            [
                np.random.permutation(
                    [0] * (n_forced_per_block // 2) + [1] * (n_forced_per_block // 2)
                )
                for _ in range(n_blocks)
            ]
        )
    else:
        forced_actions = np.zeros(T)

    return [rewards, complete, idx_blocks, choices, forced_actions]


# Simulate multiple agents
def _simulate_single_agent(args):
    (
        agent,
        model_name,
        sample,
        experiment,
        n_blocks,
        n_trials_per_block,
        apply_rep_bias,
        apply_weber,
        apply_observational_noise,
        apply_counterfactual_learn,
    ) = args

    # --- Task setup ---
    rewards, complete, idx_blocks, choices, forced_actions = task_setup(
        experiment, n_blocks, n_trials_per_block
    )

    # --- Parameters ---
    alpha_c = sample[0]
    alpha_u = sample[1] if apply_counterfactual_learn else 0
    beta = sample[2]
    epsilon = sample[3] if model_name == "noisy_rl" else 0
    rep_bias = sample[4] if apply_rep_bias else 0

    # --- Simulation function ---
    sim_fun = simulate_noisy_rl if model_name == "noisy_rl" else simulate_noiseless_rl

    # --- RUN ---
    trajectories, actions, obs_rewards, greedy_choices = sim_fun(
        rewards,
        complete,
        idx_blocks,
        choices,
        forced_actions,
        sample,
        apply_rep_bias,
        apply_weber,
        apply_observational_noise,
        apply_counterfactual_learn,
    )

    # --- Create dataframe ---
    return pd.DataFrame(
        {
            "subj_code": agent,
            "trial": np.tile(np.arange(n_trials_per_block), n_blocks),
            "block": np.repeat(np.arange(n_blocks), n_trials_per_block),
            "reward0": rewards[0],
            "reward1": rewards[1],
            "feedback_type": complete,
            "free_choice": choices,
            "action": actions,
            "greedy_choice": greedy_choices,
            "obs_reward0": obs_rewards[0],
            "obs_reward1": obs_rewards[1],
            "q_value0": trajectories[:, 0],
            "q_value1": trajectories[:, 1],
            "alpha_c": alpha_c,
            "alpha_u": alpha_u,
            "beta": beta,
            "epsilon": epsilon,
            "rep_bias": rep_bias,
        }
    )


def simulate_agents(
    model_name,
    param_sets,
    experiment,
    n_blocks,
    n_trials_per_block,
    n_agents,
    apply_rep_bias,
    apply_weber,
    apply_observational_noise,
    apply_counterfactual_learn,
    max_workers=8,
):

    sim_df = pd.DataFrame(
        columns=[
            "subj_code",
            "trial",
            "block",
            "reward0",
            "reward1",
            "feedback_type",
            "free_choice",
            "action",
            "greedy_choice",
            "obs_reward0",
            "obs_reward1",
            "q_value0",
            "q_value1",
            "alpha_c",
            "alpha_u",
            "beta",
            "epsilon",
            "rep_bias",
        ]
    )

    assert model_name in [
        "noisy_rl",
        "noiseless_rl",
    ], "Invalid model name. Choose 'noisy_rl' or 'noiseless_rl'."
    worker_args = [
        (
            agent,
            model_name,
            param_sets[i],
            experiment,
            n_blocks,
            n_trials_per_block,
            apply_rep_bias,
            apply_weber,
            apply_observational_noise,
            apply_counterfactual_learn,
        )
        for i, agent in enumerate(range(n_agents))
    ]

    if (max_workers is not None) and (max_workers > 1) and (n_agents > 1):
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            sim_subj_dfs = list(executor.map(_simulate_single_agent, worker_args))
    else:
        sim_subj_dfs = [_simulate_single_agent(args) for args in worker_args]

    sim_df = pd.concat(sim_subj_dfs, axis=0, ignore_index=True)
    sim_df = sim_df.sort_values(["subj_code", "block", "trial"]).reset_index(drop=True)

    # Add useful variables
    sim_df["optimal_choice"] = (
        sim_df.action
        == sim_df[["reward0", "reward1"]].idxmax(axis=1).str[-1].astype(int)
    ).astype(int)
    sim_df["prev_action"] = sim_df.groupby("subj_code")["action"].shift(1)
    sim_df["prev_optimal_choice"] = sim_df.groupby("subj_code")["optimal_choice"].shift(
        1
    )
    sim_df["stay"] = np.where(
        sim_df["subj_code"].eq(sim_df["subj_code"].shift(1)),
        (sim_df["prev_action"].eq(sim_df["action"])).astype(int),
        np.nan,
    ).astype(float)

    sim_df["prev_reward_val"] = np.where(
        sim_df["subj_code"].eq(sim_df["subj_code"].shift(1)),
        np.where(
            sim_df["prev_action"] == 0,
            sim_df["reward0"].shift(1),
            sim_df["reward1"].shift(1),
        ),
        np.nan,
    )

    sim_df["reward_diff"] = sim_df.reward1 - sim_df.reward0
    sim_df["prev_reward_diff"] = sim_df.groupby("subj_code")["reward_diff"].shift(1)

    sim_df["observed_reward_diff"] = sim_df.obs_reward1 - sim_df.obs_reward0
    sim_df["prev_observed_reward_diff"] = sim_df.groupby("subj_code")[
        "observed_reward_diff"
    ].shift(1)

    sim_df["q_diff"] = sim_df.q_value1 - sim_df.q_value0
    sim_df["prev_q_diff"] = sim_df.groupby("subj_code")["q_diff"].shift(1)

    sim_df["true_best"] = (sim_df["reward1"] > sim_df["reward0"]).astype(int)
    sim_df["is_switch"] = (
        sim_df.groupby(["subj_code", "block"])["true_best"]
        .diff()
        .fillna(0)
        .abs()
        .gt(0)
        .astype(int)
    )
    sim_df["trials_since_switch"] = sim_df.groupby(["subj_code", "block"])[
        "is_switch"
    ].cumsum()
    sim_df["trial_post_switch"] = sim_df.groupby(
        ["subj_code", "block", "trials_since_switch"]
    ).cumcount()

    for col in sim_df.columns:
        sim_df[col] = sim_df[col].astype(float)

    sim_df = sim_df.sort_values(["subj_code", "block", "trial"]).reset_index(drop=True)

    return sim_df


def create_uniform_prior_funcs(param_bounds_dict):
    """Create uniform prior functions from parameter bounds.

    Args:
        param_bounds_dict: Dictionary with parameter names as keys and (lower, upper) bounds as values

    Returns:
        Dictionary mapping parameter names to functions that return log prior probability
    """
    prior_funcs = {}
    for param, (lower, upper) in param_bounds_dict.items():
        width = upper - lower
        log_prior_const = -np.log(width)
        prior_funcs[param] = partial(_constant_log_prior, const=log_prior_const)
    return prior_funcs


def _constant_log_prior(x, const):
    """Return a constant log prior value (uniform prior within bounds)."""
    return const


def _process_single_agent_map(args):
    """Run MAP fitting for a single subject and return one result row.

    Args:
        args: Tuple containing
            (subj_code, data, model_name, param_bounds_dict, prior_funcs,
             apply_rep_bias, apply_weber, apply_observational_noise,
             apply_counterfactual_learn, beta_multiplier, max_iterations,
             parameters_to_fit)

    Returns:
        Dictionary with subj_code, nllh, and fit_* parameter keys, or None on failure.
    """
    (
        subj_code,
        data,
        model_name,
        param_bounds_dict,
        prior_funcs,
        apply_rep_bias,
        apply_weber,
        apply_observational_noise,
        apply_counterfactual_learn,
        beta_multiplier,
        max_iterations,
        parameters_to_fit,
    ) = args

    assert model_name in [
        "noisy_rl",
        "noiseless_rl",
    ], f"Invalid model_name: {model_name}"
    nllh_func = eval(f"nllh_{model_name}")
    is_noisy = model_name == "noisy_rl"

    # Determine active parameters based on model settings
    active_params = ["alpha_c", "beta"]
    if apply_counterfactual_learn:
        active_params.insert(1, "alpha_u")
    if is_noisy:
        active_params.append("epsilon")
    if apply_rep_bias:
        active_params.append("rep_bias")

    # Get bounds for active parameters only
    bounds = [param_bounds_dict[param] for param in active_params]

    def construct_sample(active_params_list):
        sample = np.zeros(5)
        sample[0] = active_params_list[0]
        idx = 1

        if apply_counterfactual_learn:
            sample[1] = active_params_list[idx]
            idx += 1
        else:
            sample[1] = 0

        sample[2] = active_params_list[idx]
        idx += 1

        if is_noisy:
            sample[3] = active_params_list[idx]
            idx += 1
        else:
            sample[3] = 0

        if apply_rep_bias:
            sample[4] = active_params_list[idx]
        else:
            sample[4] = 0

        return sample

    try:
        init_params = [random.uniform(l, h) for l, h in bounds]
        sub_data = data[data["subj_code"] == subj_code].copy()

        def log_prior(x):
            log_prob = 0.0
            for param, value in zip(active_params, x):
                log_prob += prior_funcs[param](value)
            return log_prob

        def neg_log_posterior(x):
            sample = construct_sample(x)
            nllh = nllh_func(
                sample,
                sub_data,
                apply_rep_bias,
                apply_weber,
                apply_observational_noise,
                apply_counterfactual_learn,
                beta_multiplier,
            )
            return nllh - log_prior(x)

        res = minimize(
            neg_log_posterior,
            init_params,
            bounds=bounds,
            method="L-BFGS-B",
            options={"maxiter": max_iterations},
        )

        final_sample = construct_sample(res.x)
        result_dict = {
            "alpha_c": final_sample[0],
            "alpha_u": final_sample[1],
            "beta": final_sample[2],
            "epsilon": final_sample[3],
            "rep_bias": final_sample[4],
        }
        nllh = res.fun
    except Exception as e:
        print(f"Error processing subject {subj_code}: {str(e)}")
        return None

    subj_res = {"subj_code": subj_code, "nllh": float(nllh)}
    subj_res.update(
        {
            f"fit_{k}": float(v)
            for k, v in result_dict.items()
            if (parameters_to_fit is None) or (k in parameters_to_fit)
        }
    )
    return subj_res


def process_agents_map(
    data,
    model_name,
    param_bounds_dict,
    prior_funcs,
    apply_rep_bias,
    apply_weber,
    apply_observational_noise,
    apply_counterfactual_learn,
    beta_multiplier=20,
    max_iterations=30,
    parameters_to_fit=None,
    max_workers=8,
):
    """Wrapper that fits MAP parameters for multiple subjects.

    This function organizes worker arguments and runs them in a process pool,
    following the same structure as simulate_agents/_simulate_single_agent.

    Args:
        data: DataFrame containing one or more subjects with a subj_code column.
        model_name: Model name ("noisy_rl" or "noiseless_rl").
        param_bounds_dict: Bounds for MAP fitting.
        prior_funcs: Prior functions keyed by parameter name.
        apply_rep_bias: Whether rep_bias is active.
        apply_weber: Whether Weber scaling is active.
        apply_observational_noise: Whether observational noise is active.
        apply_counterfactual_learn: Whether alpha_u is active.
        beta_multiplier: Multiplier used for beta inside likelihood.
        max_iterations: Optimizer max iterations.
        parameters_to_fit: Optional list of parameters to keep in output.
        max_workers: Number of worker processes; use 1 for sequential execution.

    Returns:
        DataFrame with true and fitted parameters per subject.
    """
    subj_codes = data["subj_code"].unique()
    worker_args = [
        (
            subj_code,
            data,
            model_name,
            param_bounds_dict,
            prior_funcs,
            apply_rep_bias,
            apply_weber,
            apply_observational_noise,
            apply_counterfactual_learn,
            beta_multiplier,
            max_iterations,
            parameters_to_fit,
        )
        for subj_code in subj_codes
    ]

    if (max_workers is not None) and (max_workers > 1) and (len(worker_args) > 1):
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            param_rows = list(executor.map(_process_single_agent_map, worker_args))
    else:
        param_rows = [_process_single_agent_map(args) for args in worker_args]

    param_rows = [row for row in param_rows if row is not None]
    fit_res = pd.DataFrame(param_rows)

    if parameters_to_fit is None:
        parameters_to_fit = list(param_bounds_dict.keys())

    for p in parameters_to_fit:
        fit_col = f"fit_{p}"
        if fit_col not in fit_res.columns:
            fit_res[fit_col] = np.nan

    if fit_res.empty:
        fit_res = pd.DataFrame(
            columns=["subj_code", "nllh"] + [f"fit_{p}" for p in parameters_to_fit]
        )
    else:
        fit_res = fit_res[
            ["subj_code", "nllh"] + [f"fit_{p}" for p in parameters_to_fit]
        ]

    fit_res = (
        data.groupby("subj_code")[parameters_to_fit]
        .first()
        .reset_index()
        .merge(fit_res, on="subj_code", how="left")
        .convert_dtypes()
    )
    return fit_res
