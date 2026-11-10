"""RLF parameter estimation (Fig. 8B and Table A2): train and test the GRU on all 78 runs.

A run is a training file, a budget (agents drawn from that file), a subsample seed (which agents the budget draws),
a feature set and an init seed. runs.csv lists the 60 runs of Fig. 8B (the six cumulative feature sets at 1,000,
3,000 and 9,000 agents drawn three times from the 27,000-agent file, and all 27,000 agents once) and the 18 runs of
Table A2 (the six features with one swapped or removed, the 9,000-agent file, three init seeds), each with its
configuration (configurations.csv). The script reads data/ and writes to results/parameter_estimation/rlf/.

    python training/local/parameter_estimation/rlf/rlf.py
"""

import argparse
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import numpy as np
import pandas as pd

REPO_DIR = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_DIR))

import tensorflow as tf
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import EarlyStopping

from training.models import get_model_for_parameter_estimation
from utils.feature_utils import (
    extract_features_blockless,
    get_condition_aware_labels,
    get_iter_acc_with_switches,
    get_iter_acc_without_switches,
    get_recovered_parameters,
    normalize_train_labels,
    normalize_val_labels,
)
from utils.model_utils import set_seed

HERE = Path(__file__).resolve().parent
DATA_DIR = REPO_DIR / "data"
OUT_DIR = REPO_DIR / "results" / "parameter_estimation" / "rlf"
# The test file is in data/; the train files are simulated with simulation/simulation.ipynb.
DATA_FILES = {
    "9000a": "rlf_6s3a/rand_500seqs_9000a_12b_78t_rlf.csv",
    "27000a": "rlf_6s3a/rand_500seqs_27000a_12b_78t_rlf.csv",
}
TEST_FILE = "rlf_6s3a/rand500_st11_500a_12b_78t_rlf.csv"
# Fig. 8B: the features in order of importance, added one at a time.
CUMULATIVE_SETS = {
    "1:rewards": ["rewards"],
    "2:+actions": ["rewards", "actions"],
    "3:+prev_action": ["rewards", "actions", "prev_action"],
    "4:+stim_iter": ["rewards", "actions", "prev_action", "stim_iter"],
    "5:+delay_stim": ["rewards", "actions", "prev_action", "stim_iter", "delay_since_last_stimuli"],
    "6:+isswitch": ["rewards", "actions", "prev_action", "stim_iter", "delay_since_last_stimuli", "isswitch"],
}
# Table A2: the six features with one swapped or removed.
ABLATION_SETS = {
    "orig": ["rewards", "actions", "prev_action", "stim_iter", "delay_since_last_stimuli", "isswitch"],
    "switch2block": ["rewards", "actions", "prev_action", "stim_iter", "delay_since_last_stimuli", "block_no"],
    "switch2trial": ["rewards", "actions", "prev_action", "stim_iter", "delay_since_last_stimuli", "trials"],
    "noswitch": ["rewards", "actions", "prev_action", "stim_iter", "delay_since_last_stimuli"],
    "stim2stimuli_nosort": ["rewards", "actions", "prev_action", "stimuli", "delay_since_last_stimuli", "isswitch"],
    "noprev": ["rewards", "actions", "stim_iter", "delay_since_last_stimuli", "isswitch"],
}
FEATURE_SETS = {**CUMULATIVE_SETS, **ABLATION_SETS}
# Sorted by stimulus, a stimulus' presentations are adjacent and stim_iter counts them; the stimulus id itself is
# kept in trial order.
SORTED_COL = ["agentid", "block_no", "stimuli"]
UNSORTED_SETS = ["stim2stimuli_nosort"]
TARGETED_PARAMS = ["alpha", "beta", "phi", "stickiness"]
BETA_MULTIPLIER = 20
EPOCHS = 600
PATIENCE = 50


def read_data(path: Path) -> pd.DataFrame:
    """A simulated file, rows in trial order within each agent and block."""
    return pd.read_csv(path).sort_values(["agentid", "block_no", "trials"]).reset_index(drop=True)


def get_iteration_features(df: pd.DataFrame) -> pd.DataFrame:
    """The workflow's input features: stim_iter, delay_since_last_stimuli, isswitch and prev_action."""
    has_switches = df.isswitch.nunique() == 2
    df_iter = get_iter_acc_with_switches(df) if has_switches else get_iter_acc_without_switches(df)
    df_iter["delay_since_last_stimuli"] = df["delay_since_last_stimuli"].to_list()

    if has_switches:
        df_iter.loc[df_iter.trials == 0, "isswitch"] = 1
    else:
        df_iter["isswitch"] = df_iter["trials"].apply(lambda x: 1 if x == 0 else 0)

    df_iter["prev_action"] = (
        df.groupby(["agentid", "block_no"])["actions"]
        .shift(1)
        .fillna(-1)
        .astype(int)
        .to_list()
    )
    return df_iter


def get_labels(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """One label per agent and parameter, beta divided by the simulation's multiplier."""
    labels = get_condition_aware_labels(df, TARGETED_PARAMS, [])
    labels["beta"] = labels["beta"] / BETA_MULTIPLIER
    return labels


def prepare_dataset(train_df: pd.DataFrame, budget: int, seed: Optional[int], sorts: Set[bool]) -> Dict:
    """The budget's agents: input features (sorted and in trial order) and normalized labels, training and validation.

    A budget smaller than the file draws its agents with the subsample seed; 10% of the agents go to validation.
    """
    agent_ids = train_df.agentid.unique()
    if budget < len(agent_ids):
        drawn = np.random.RandomState(seed).choice(agent_ids, size=budget, replace=False)
        train_df = train_df[train_df.agentid.isin(set(drawn))]

    val_agent_ids = np.random.RandomState(42).choice(
        train_df.agentid.unique(), int(train_df.agentid.nunique() * 0.1), replace=False
    )
    train_labels = get_labels(train_df[~train_df.agentid.isin(val_agent_ids)])
    val_labels = get_labels(train_df[train_df.agentid.isin(val_agent_ids)])
    normalized_train_labels, name_to_scaler = normalize_train_labels(train_labels)
    normalized_val_labels = normalize_val_labels(val_labels, name_to_scaler)

    td_iter = get_iteration_features(train_df)
    frames = {}
    for sort in sorts:
        td_iter_order = td_iter.sort_values(SORTED_COL) if sort else td_iter
        is_val = td_iter_order.agentid.isin(val_agent_ids)
        frames[sort] = (td_iter_order[~is_val], td_iter_order[is_val])

    return {
        "n_agents": train_df.agentid.nunique(),
        "frames": frames,
        "normalized_train_labels": normalized_train_labels,
        "normalized_val_labels": normalized_val_labels,
        "name_to_scaler": name_to_scaler,
    }


def prepare_test(test_df: pd.DataFrame) -> Dict:
    """The 500 test agents: input features (sorted and in trial order) and true labels."""
    test_iter = get_iteration_features(test_df)
    return {
        "agent_ids": np.sort(test_df.agentid.unique()),
        "frames": {True: test_iter.sort_values(SORTED_COL), False: test_iter},
        "true_labels": get_labels(test_df),
    }


def train_run(
    dataset: Dict, test: Dict, run: pd.Series, configuration: pd.Series
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, tf.keras.Model]:
    """Train and test one GRU: one row per parameter, one row per test agent, the loss curves and the model."""
    feature_list = FEATURE_SETS[run.feature_set]
    sort = run.feature_set not in UNSORTED_SETS
    training_data, val_data = dataset["frames"][sort]
    train_features = extract_features_blockless(training_data, feature_list, [])
    val_features = extract_features_blockless(val_data, feature_list, [])
    train_features, normalized_train_labels = shuffle(
        train_features, dataset["normalized_train_labels"], random_state=0
    )
    val_features, normalized_val_labels = shuffle(val_features, dataset["normalized_val_labels"], random_state=0)

    tf.keras.backend.clear_session()
    set_seed(int(run.init_seed))
    model = get_model_for_parameter_estimation(
        {
            "feature_dim": train_features.shape[-1],
            "output_dims": normalized_train_labels.shape[-1],
            "units": int(configuration.units),
            "dropout": float(configuration.dropout),
            "dropout1": float(configuration.dropout1),
            "dropout2": float(configuration.dropout2),
            "learning_rate": float(configuration.learning_rate),
        }
    )
    start = time.time()
    history = model.fit(
        train_features,
        normalized_train_labels,
        epochs=EPOCHS,
        batch_size=int(configuration.batch_size),
        callbacks=[EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)],
        validation_data=(val_features, normalized_val_labels),
        verbose=2,
    )
    train_minutes = (time.time() - start) / 60

    test_features = extract_features_blockless(test["frames"][sort], feature_list, [])
    predicted = model.predict(test_features, verbose=0)
    result = get_recovered_parameters(dataset["name_to_scaler"], test["true_labels"], predicted)

    run_info = {
        "run_tag": run.run_tag,
        "budget": dataset["n_agents"],
        "seed": run.seed,
        "init_seed": int(run.init_seed),
        "feature_set": run.feature_set,
        "n_features": len(feature_list),
    }
    per_agent = pd.DataFrame({**run_info, "id": test["agent_ids"]})
    for p in TARGETED_PARAMS:
        per_agent[f"true_{p}"] = result[f"true_{p}"].to_numpy()
        per_agent[f"pred_{p}"] = result[f"dl_{p}"].to_numpy()

    summary = pd.DataFrame(
        [
            {
                **run_info,
                "configuration": run.configuration,
                "parameter": p,
                "rmse": root_mean_squared_error(result[f"true_{p}"], result[f"dl_{p}"]),
                "mae": mean_absolute_error(result[f"true_{p}"], result[f"dl_{p}"]),
                "r": result[f"true_{p}"].corr(result[f"dl_{p}"]),
                "val_loss": float(np.min(history.history["val_loss"])),
                "epochs_ran": len(history.history["loss"]),
                "train_minutes": train_minutes,
            }
            for p in TARGETED_PARAMS
        ]
    )
    return summary, per_agent, pd.DataFrame(history.history), model


def ablation_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Table A2: each model's RMSE averaged over the four parameters, then the mean and SD over its init seeds."""
    per_model = (
        summary.groupby(["feature_set", "run_tag"])
        .agg(rmse=("rmse", "mean"), train_minutes=("train_minutes", "first"))
        .reset_index()
    )
    table = per_model.groupby("feature_set").agg(
        n_runs=("run_tag", "size"),
        rmse_mean=("rmse", "mean"),
        rmse_std=("rmse", "std"),
        minutes_mean=("train_minutes", "mean"),
        minutes_std=("train_minutes", "std"),
    )
    return table.reindex([s for s in ABLATION_SETS if s in table.index]).reset_index()


def write_results(out_dir: Path, summaries: list, per_agents: list) -> None:
    """Fig. 8B and Table A2 in separate files, rewritten after every run."""
    summary = pd.concat(summaries, ignore_index=True)
    per_agent = pd.concat(per_agents, ignore_index=True)
    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    is_cumulative = per_agent.feature_set.isin(list(CUMULATIVE_SETS))
    if is_cumulative.any():
        per_agent[is_cumulative].to_csv(os.path.join(out_dir, "recovery_per_agent.csv"), index=False)
    if not is_cumulative.all():
        per_agent[~is_cumulative].to_csv(os.path.join(out_dir, "ablation_per_agent.csv"), index=False)
        ablation = summary[summary.feature_set.isin(list(ABLATION_SETS))]
        ablation_table(ablation).to_csv(os.path.join(out_dir, "ablation_table.csv"), index=False)


def main():
    # No options: every run of runs.csv is trained. --help prints the docstring.
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()

    runs = pd.read_csv(HERE / "runs.csv", dtype={"seed": "Int64"})
    configurations = pd.read_csv(HERE / "configurations.csv").set_index("configuration")
    print(f"{len(runs)} runs, results in {OUT_DIR}", flush=True)
    os.makedirs(OUT_DIR / "models", exist_ok=True)

    test = prepare_test(read_data(DATA_DIR / TEST_FILE))
    summaries, per_agents = [], []
    for train_file, file_runs in runs.groupby("train_file", sort=False):
        train_df = read_data(DATA_DIR / DATA_FILES[train_file])
        # The feature sets of a budget and seed share one dataset.
        for (budget, seed), group in file_runs.groupby(["budget", "seed"], sort=False, dropna=False):
            sorts = {feature_set not in UNSORTED_SETS for feature_set in group.feature_set}
            dataset = prepare_dataset(train_df, int(budget), None if pd.isna(seed) else int(seed), sorts)
            for _, run in group.iterrows():
                print(f"{run.run_tag}: configuration {run.configuration}", flush=True)
                summary, per_agent, loss, model = train_run(
                    dataset, test, run, configurations.loc[run.configuration]
                )
                loss.to_csv(OUT_DIR / f"{run.run_tag}_loss.csv", index=False)
                model.save(str(OUT_DIR / "models" / f"{run.run_tag}.keras"))
                with open(OUT_DIR / "models" / f"{run.run_tag}_label_scaler.pkl", "wb") as f:
                    pickle.dump(dataset["name_to_scaler"], f)
                summaries.append(summary)
                per_agents.append(per_agent)
                write_results(OUT_DIR, summaries, per_agents)
                print(summary.to_string(index=False), flush=True)
        del train_df


if __name__ == "__main__":
    main()
