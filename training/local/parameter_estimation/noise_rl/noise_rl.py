"""Noisy RL parameter estimation (Table B2): train and test the GRU on all 72 runs of the appendix.

A run is a training budget (3,000 or 9,000 agents), a feature set (FEATURE_SETS), an init seed and a configuration
(configurations.csv). runs.csv lists the 72 runs: the `default` configuration for the 18 models of Table B2 and the
same 18 at 9,000 agents, and for the 36 tuned models the winner of the search of their budget and feature set.
The script reads data/ and writes to results/parameter_estimation/noise_rl/.

    python training/local/parameter_estimation/noise_rl/noise_rl.py
"""

import argparse
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

REPO_DIR = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_DIR))

import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import EarlyStopping

from training.models import get_model_for_parameter_estimation
from utils.feature_importance_utils import permutation_importance_global_sequence
from utils.feature_utils import (
    extract_features_blockless,
    get_labels,
    get_recovered_parameters,
    normalize_train_labels,
    normalize_val_labels,
)
from utils.model_utils import set_seed

HERE = Path(__file__).resolve().parent
DATA_DIR = REPO_DIR / "data"
OUT_DIR = REPO_DIR / "results" / "parameter_estimation" / "noise_rl"
# The three files are simulated with simulation/noise_rl.py; the README has the calls.
FILE_NAME = "noisyrl/sim_data_noisy_rl_exp3_nAgents{}_beta35_alphaMax0.8_repBias1_weber1_obsNoise1_cfLearn1.csv.gz"
TRAIN_FILES = {3000: FILE_NAME.format(3000), 9000: FILE_NAME.format(9000)}
TEST_FILE = FILE_NAME.format(500)
# The seed each training population was simulated with; it also draws the population's validation agents.
SPLIT_SEEDS = {3000: 20260916, 9000: 20260917}
PARAMETERS = ["alpha_c", "alpha_u", "epsilon", "xi"]
# The input order of the network.
FEATURE_LIST = [
    "actions",
    "prev_action",
    "block_start",
    "optimal_choice",
    "chosen_reward_bin",
    "unchosen_reward_bin",
    "chosen_vs_unchosen_reward_bin",
    "reward0_bin",
    "reward1_bin",
]
# The base set, then the other five features added one at a time (the rows of Table B2); each set keeps the order of
# FEATURE_LIST.
BASE = ["actions", "prev_action", "chosen_vs_unchosen_reward_bin", "reward1_bin"]
ADDED = ["reward0_bin", "chosen_reward_bin", "unchosen_reward_bin", "block_start", "optimal_choice"]
FEATURE_SETS = {
    name: [f for f in FEATURE_LIST if f in BASE + ADDED[:i]]
    for i, name in enumerate(["base4", "five", "six", "seven", "no_optimal", "all9"])
}
EPOCHS = 600
PATIENCE = 50


def preprocess(df: pd.DataFrame, max_trials: float) -> pd.DataFrame:
    """Rows sorted by agent, block and trial; the features, with the payoffs cut into five equal bins; xi."""
    df = df.sort_values(["subj_code", "block", "trial"]).reset_index(drop=True)
    df["block_start"] = (df.groupby("subj_code")["block"].diff() != 0).astype(int)
    df["prev_action"] = df.groupby("subj_code")["action"].shift(1).fillna(-1).astype(int)

    df["chosen_reward"] = np.where(df.action == 0, df.reward0, df.reward1)
    df["unchosen_reward"] = np.where(df.action == 1, df.reward0, df.reward1)
    df["chosen_vs_unchosen_reward"] = df["chosen_reward"] - df["unchosen_reward"]

    df["xi"] = df.rep_bias.copy()
    df["trial_n"] = df["trial"] / max_trials

    n_bins = 5
    for col in ["chosen_reward", "unchosen_reward", "chosen_vs_unchosen_reward", "reward0", "reward1"]:
        min_val, max_val = [[0, 1], [-1, 1]][col == "chosen_vs_unchosen_reward"]
        df[f"{col}_bin"] = pd.cut(df[col], bins=np.linspace(min_val, max_val, n_bins + 1), labels=np.arange(n_bins))

    return df.rename(columns={"subj_code": "agentid", "block": "block_no", "action": "actions", "trial_n": "trials"})


def prepare_dataset(data_dir: str, budget: int) -> Dict:
    """Features of all nine inputs and standardized labels: train, validation and test.

    A tenth of the training agents validate, drawn with the population's seed; the labels are standardized with the
    mean and standard deviation of the remaining training agents.
    """
    train = pd.read_csv(os.path.join(data_dir, TRAIN_FILES[budget]))
    test = pd.read_csv(os.path.join(data_dir, TEST_FILE))
    max_trials = train["trial"].astype(float).max()
    train, test = preprocess(train, max_trials), preprocess(test, max_trials)

    rng = np.random.RandomState(SPLIT_SEEDS[budget])
    val_agent_ids = set(rng.choice(train.agentid.unique(), int(budget * 0.1), replace=False))
    val = train[train.agentid.isin(val_agent_ids)].reset_index(drop=True)
    train = train[~train.agentid.isin(val_agent_ids)].reset_index(drop=True)

    test_labels = get_labels(test, PARAMETERS)
    y_train, name_to_scaler = normalize_train_labels(get_labels(train, PARAMETERS))
    return {
        "X_train": extract_features_blockless(train, FEATURE_LIST, []),
        "X_val": extract_features_blockless(val, FEATURE_LIST, []),
        "X_test": extract_features_blockless(test, FEATURE_LIST, []),
        "y_train": y_train,
        "y_val": normalize_val_labels(get_labels(val, PARAMETERS), name_to_scaler),
        "y_test": normalize_val_labels(test_labels, name_to_scaler),
        "test_labels": test_labels,
        "test_ids": np.sort(test.agentid.unique()),
        "name_to_scaler": name_to_scaler,
    }


def train_run(
    dataset: Dict, run: pd.Series, configuration: pd.Series
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, tf.keras.Model]:
    """Train and test one model: its summary rows, one row per test agent, its feature importance, the loss curves
    and the model."""
    features = FEATURE_SETS[run.feature_set]
    columns = [FEATURE_LIST.index(f) for f in features]
    X_train, X_val, X_test = (dataset[x][:, :, columns] for x in ["X_train", "X_val", "X_test"])
    X_train, y_train = shuffle(X_train, dataset["y_train"], random_state=0)
    X_val, y_val = shuffle(X_val, dataset["y_val"], random_state=0)

    tf.keras.backend.clear_session()
    set_seed(int(run.init_seed))
    model = get_model_for_parameter_estimation(
        {
            "feature_dim": len(features),
            "output_dims": len(PARAMETERS),
            "units": int(configuration.units),
            "dropout": float(configuration.dropout),
            "dropout1": float(configuration.dropout1),
            "dropout2": float(configuration.dropout2),
            "learning_rate": float(configuration.learning_rate),
        }
    )
    start = time.time()
    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=int(configuration.batch_size),
        callbacks=[EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)],
        validation_data=(X_val, y_val),
        verbose=2,
    )
    train_minutes = (time.time() - start) / 60

    recovered = get_recovered_parameters(
        dataset["name_to_scaler"], dataset["test_labels"], model.predict(X_test, verbose=0)
    )
    run_info = {
        "model": run.run_tag,
        "n_train_agents": int(run.budget),
        "n_features": len(features),
        "features": " ".join(features),
        "init_seed": int(run.init_seed),
        "tuned": run.configuration != "default",
    }
    per_agent = recovered.rename(columns=lambda c: c.replace("dl_", "pred_"))
    per_agent.insert(0, "agentid", dataset["test_ids"])
    per_agent = per_agent.assign(**run_info)

    val_loss = np.array(history.history["val_loss"])
    training_info = {
        "units": int(configuration.units),
        "batch_size": int(configuration.batch_size),
        "learning_rate": float(configuration.learning_rate),
        "epochs_ran": len(val_loss),
        "best_epoch": int(np.argmin(val_loss)) + 1,
        "best_val_loss": float(val_loss.min()),
        "train_minutes": round(train_minutes, 2),
    }
    summary = pd.DataFrame(
        [
            {
                "parameter": p,
                "rmse": float(np.sqrt(np.mean((per_agent[f"pred_{p}"] - per_agent[f"true_{p}"]) ** 2))),
                "r": float(np.corrcoef(per_agent[f"true_{p}"], per_agent[f"pred_{p}"])[0, 1]),
                "true_min": float(per_agent[f"true_{p}"].min()),
                "true_max": float(per_agent[f"true_{p}"].max()),
                **run_info,
                **training_info,
            }
            for p in sorted(PARAMETERS)
        ]
    )

    importance = permutation_importance_global_sequence(
        model=model,
        X_val=X_test,
        y_val=dataset["y_test"],
        feature_names=features,
        n_repeats=3,
        batch_size=64,
        shuffle_mode="within_sample",
        block_size=32,
        random_state=123,
    ).assign(**run_info, **training_info)
    return summary, per_agent, importance, pd.DataFrame(history.history), model


def main():
    # No options: every run of runs.csv is trained. --help prints the docstring.
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()

    runs = pd.read_csv(HERE / "runs.csv")
    configurations = pd.read_csv(HERE / "configurations.csv").set_index("configuration")
    print(f"{len(runs)} runs, results in {OUT_DIR}", flush=True)
    os.makedirs(OUT_DIR / "models", exist_ok=True)

    tables = {"recovery_summary": [], "recovery_per_agent": [], "feature_importance": []}
    # The runs of a budget share one dataset.
    for budget, group in runs.groupby("budget", sort=False):
        dataset = prepare_dataset(DATA_DIR, int(budget))
        for _, run in group.iterrows():
            print(f"{run.run_tag}: configuration {run.configuration}", flush=True)
            summary, per_agent, importance, loss, model = train_run(
                dataset, run, configurations.loc[run.configuration]
            )
            loss.to_csv(OUT_DIR / f"{run.run_tag}_loss.csv", index=False)
            model.save(str(OUT_DIR / "models" / f"{run.run_tag}.keras"))
            with open(OUT_DIR / "models" / f"{run.run_tag}_label_scaler.pkl", "wb") as fh:
                pickle.dump(dataset["name_to_scaler"], fh)
            # The tables are rewritten after every run, so an interrupted job keeps the runs it has finished.
            for name, table in zip(tables, [summary, per_agent, importance]):
                tables[name].append(table)
                pd.concat(tables[name]).to_csv(OUT_DIR / f"{name}.csv", index=False)
            print(summary[["parameter", "rmse", "r"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
