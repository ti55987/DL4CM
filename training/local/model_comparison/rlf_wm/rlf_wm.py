"""RLF vs WM model comparison (Fig. 8C): train and test the GRU classifier on all 50 runs of the figure.

A run is a feature set (the first k features of FEATURE_LIST, k = 1, 2, 3, 4 or 6), a simulation budget (agents
per model) and a subsample seed (which agents the budget draws). runs.csv lists the 50 runs of Fig. 8C with the
init seed and the configuration (configurations.csv) of each reported model. The script reads data/ and writes to
results/model_comparison/rlf_wm/.

    python training/local/model_comparison/rlf_wm/rlf_wm.py
"""

import argparse
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

REPO_DIR = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_DIR))

import tensorflow as tf
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import EarlyStopping

from training.models import get_model_for_model_identification
from utils.feature_utils import extract_features_blockless
from utils.model_utils import set_seed

HERE = Path(__file__).resolve().parent
DATA_DIR = REPO_DIR / "data"
OUT_DIR = REPO_DIR / "results" / "model_comparison" / "rlf_wm"
MODELS = ["RLF", "WM"]  # class 0 and class 1
# The test files are in data/; the train files are simulated with simulation/simulation.ipynb.
DATA_FILES = {
    "RLF": {
        "train": "rlf_6s3a/rand_500seqs_27000a_12b_78t_rlf.csv",
        "test": "rlf_6s3a/rand500_st11_500a_12b_78t_rlf.csv",
    },
    "WM": {
        "train": "wm_6s3a/rand500_st11_27000a_12b_78t_wm.csv",
        "test": "wm_6s3a/rand500_st11_500a_12b_78t_wm.csv",
    },
}
# Most important first; the feature sets of Fig. 8C are the first 1, 2, 3, 4 and 6.
FEATURE_LIST = [
    "prev_action",
    "rewards",
    "actions",
    "delay_since_last_stimuli",
    "isswitch",
    "stim_iter",
]
SORTED_COL = ["agentid", "block_no", "stimuli"]
EPOCHS = 600
PATIENCE = 50
# A model whose best validation loss stays at ln 2 (chance for two classes) never trained.
DEAD_VAL_LOSS = 0.690

# float32 softmax outputs miss a sum of one by about 1e-7, and log_loss warns once per test agent.
warnings.filterwarnings("ignore", message="The y_prob values do not sum to one")


def read_data(data_dir: Path) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Train and test files of both models."""
    return {
        model: {
            split: pd.read_csv(os.path.join(data_dir, path))
            for split, path in files.items()
        }
        for model, files in DATA_FILES.items()
    }


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    """isswitch and prev_action, then the rows sorted by agent, block and stimulus."""
    data = data.copy()
    data["isswitch"] = (data["trials"] == 0).astype(int)
    data["prev_action"] = (
        data.groupby(["agentid", "block_no"])["actions"]
        .shift(1)
        .fillna(-1)
        .astype(int)
        .to_list()
    )
    return data.sort_values(by=SORTED_COL)


def prepare_model_dataset(
    data: Dict[str, Dict[str, pd.DataFrame]],
    budget: Optional[int] = None,
    subsample_seed: Optional[int] = None,
    test_size: float = 0.1,
    random_state: int = 42,
) -> Dict[str, np.ndarray]:
    """Features of all six inputs and labels of both models: train, validation and test.

    With a budget, subsample_seed draws which agents each model gets (the same agent ids for both models, whose
    train files share ids and stimulus sequences); without a seed the budget takes the first agents. No budget keeps
    all 27,000. Each model's agents are split into train and validation (test_size, random_state).
    """
    train_ids = None
    if budget is not None:
        pool = np.sort(data["RLF"]["train"].agentid.unique())
        train_ids = (
            np.random.default_rng(subsample_seed).choice(pool, size=budget, replace=False)
            if subsample_seed is not None
            else pool[:budget]
        )

    dataset = {k: [] for k in ["X_train", "X_val", "X_test", "y_train", "y_val", "y_test", "test_agents"]}
    for idx, model in enumerate(MODELS):
        train = data[model]["train"]
        if train_ids is not None:
            train = train[train.agentid.isin(set(train_ids))]
            if train.agentid.nunique() != budget:
                raise ValueError(f"{model}: {train.agentid.nunique()} of the {budget} drawn agents are in the file")
        features = extract_features_blockless(add_features(train), FEATURE_LIST, [])
        trf, valf, train_l, val_l = train_test_split(
            features, [idx] * len(features), test_size=test_size, random_state=random_state
        )
        test = add_features(data[model]["test"])
        dataset["X_train"].append(trf)
        dataset["X_val"].append(valf)
        dataset["X_test"].append(extract_features_blockless(test, FEATURE_LIST, []))
        dataset["y_train"].extend(train_l)
        dataset["y_val"].extend(val_l)
        dataset["y_test"].extend([idx] * test.agentid.nunique())
        dataset["test_agents"].append(pd.DataFrame({"model_type": model, "id": test.agentid.unique()}))

    return {
        "X_train": np.concatenate(dataset["X_train"]),
        "X_val": np.concatenate(dataset["X_val"]),
        "X_test": np.concatenate(dataset["X_test"]),
        "y_train": tf.keras.utils.to_categorical(dataset["y_train"], num_classes=len(MODELS)),
        "y_val": tf.keras.utils.to_categorical(dataset["y_val"], num_classes=len(MODELS)),
        "y_test": np.array(dataset["y_test"]),
        "test_agents": pd.concat(dataset["test_agents"], ignore_index=True),
    }


def train_run(
    dataset: Dict[str, np.ndarray], run: pd.Series, configuration: pd.Series
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, tf.keras.Model]:
    """Train and test one classifier: its summary row, one row per test agent, the loss curves and the model."""
    n_features = int(run.n_features)
    X_train, X_val, X_test = (dataset[x][:, :, :n_features] for x in ["X_train", "X_val", "X_test"])
    X_train, y_train = shuffle(X_train, dataset["y_train"], random_state=0)
    X_val, y_val = shuffle(X_val, dataset["y_val"], random_state=0)

    tf.keras.backend.clear_session()
    set_seed(int(run.init_seed))
    model = get_model_for_model_identification(
        {
            "feature_dim": n_features,
            "output_dims": len(MODELS),
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

    prediction = model.predict(X_test, verbose=0)
    y_true, y_pred = dataset["y_test"], np.argmax(prediction, axis=1)
    nll = [log_loss([t], [p], labels=list(range(len(MODELS)))) for t, p in zip(y_true, prediction)]
    val_loss = float(np.min(history.history["val_loss"]))
    run_info = {
        "run_tag": run.run_tag,
        "budget": int(run.budget),
        "seed": int(run.seed),
        "init_seed": int(run.init_seed),
        "dead": val_loss >= DEAD_VAL_LOSS,
        "n_features": n_features,
    }
    per_agent = pd.DataFrame(
        {
            **run_info,
            "model_type": dataset["test_agents"].model_type,
            "id": dataset["test_agents"].id,
            "y_true": y_true,
            "y_pred": y_pred,
            "p_RLF": prediction[:, 0],
            "p_WM": prediction[:, 1],
            "nll": nll,
        }
    )
    summary = pd.DataFrame(
        [
            {
                **run_info,
                "configuration": run.configuration,
                "accuracy": float((y_true == y_pred).mean()),
                "nll": float(np.mean(nll)),
                "val_loss": val_loss,
                "epochs_ran": len(history.history["loss"]),
                "train_minutes": train_minutes,
            }
        ]
    )
    return summary, per_agent, pd.DataFrame(history.history), model


def main():
    # No options: every run of runs.csv is trained. --help prints the docstring.
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()

    runs = pd.read_csv(HERE / "runs.csv")
    configurations = pd.read_csv(HERE / "configurations.csv").set_index("configuration")
    print(f"{len(runs)} runs, results in {OUT_DIR}", flush=True)
    os.makedirs(OUT_DIR / "models", exist_ok=True)

    data = read_data(DATA_DIR)
    summaries, per_agents = [], []
    # The five feature sets of a budget and seed share one dataset.
    for (budget, seed), group in runs.groupby(["budget", "seed"], sort=False):
        dataset = prepare_model_dataset(data, None if budget == 27000 else int(budget), int(seed))
        for _, run in group.iterrows():
            print(f"{run.run_tag}: configuration {run.configuration}", flush=True)
            summary, per_agent, loss, model = train_run(dataset, run, configurations.loc[run.configuration])
            loss.to_csv(OUT_DIR / f"{run.run_tag}_loss.csv", index=False)
            model.save(str(OUT_DIR / "models" / f"{run.run_tag}.keras"))
            summaries.append(summary)
            per_agents.append(per_agent)
            print(summary.to_string(index=False), flush=True)

    pd.concat(summaries).to_csv(OUT_DIR / "summary.csv", index=False)
    pd.concat(per_agents).to_csv(OUT_DIR / "identification_per_agent.csv", index=False)


if __name__ == "__main__":
    main()
