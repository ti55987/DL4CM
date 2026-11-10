"""Hyperparameter search for the RLF vs WM classifier: the two searches behind the 9k and 27k configurations.

KerasBayesianSearch, as in the workflow notebooks: ten trials of up to 35 epochs (early stopping, patience 35) on
all six features, with the first 9,000 and then all 27,000 agents per model, scored by their best validation loss.
The script reads data/ and writes search_trials_9000a.csv and search_trials_27000a.csv to
results/model_comparison/rlf_wm/, one row per trial, best first (trials whose loss went NaN last).

    python training/local/model_comparison/rlf_wm/search.py
"""

import argparse
import math
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from hyperopt import Trials, hp
from kerashypetune import KerasBayesianSearch
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import EarlyStopping

from rlf_wm import DATA_DIR, FEATURE_LIST, MODELS, OUT_DIR, prepare_model_dataset, read_data
from training.models import get_model_for_model_identification
from utils.model_utils import set_seed

BUDGETS = [9000, 27000]  # agents per model; 27000 is all of them
SAMPLING_SEED = 33
HP_KEYS = ["units", "dropout", "dropout1", "dropout2", "learning_rate", "batch_size"]


def main():
    # No options: both searches run. --help prints the docstring.
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    data = read_data(DATA_DIR)

    for budget in BUDGETS:
        dataset = prepare_model_dataset(data, None if budget == 27000 else budget)
        X_train, y_train = shuffle(dataset["X_train"], dataset["y_train"], random_state=0)
        X_val, y_val = shuffle(dataset["X_val"], dataset["y_val"], random_state=0)

        param_grid = {
            "feature_dim": len(FEATURE_LIST),
            "output_dims": len(MODELS),
            "units": hp.choice("units", [96, 128, 512]),
            "learning_rate": hp.loguniform("learning_rate", np.log(3e-4), np.log(3e-2)),
            "dropout": hp.uniform("dropout", 0.15, 0.25),
            "dropout1": hp.uniform("dropout1", 0.01, 0.1),
            "dropout2": hp.uniform("dropout2", 0.01, 0.05),
            "epochs": 35,
            "batch_size": hp.choice("batch_size", [128, 256, 512]),
        }
        tf.keras.backend.clear_session()
        set_seed(SAMPLING_SEED)
        kbs = KerasBayesianSearch(
            get_model_for_model_identification,
            param_grid,
            monitor="val_loss",
            greater_is_better=False,
            n_iter=10,
            sampling_seed=SAMPLING_SEED,
        )
        kbs.search(
            X_train,
            y_train,
            trials=Trials(),
            validation_data=(X_val, y_val),
            callbacks=[EarlyStopping(monitor="val_loss", patience=35)],
        )

        # kbs.best_params can point at a trial whose loss went NaN, so every trial is written out, the finite ones
        # first. After the search a trial's "epochs" is the epoch of its best validation loss.
        trials = pd.DataFrame(
            [
                {
                    **{k: params[k].item() if hasattr(params[k], "item") else params[k] for k in HP_KEYS},
                    "best_epoch": int(params["epochs"]),
                    "val_loss": float(score),
                    "trial": i,
                }
                for i, (params, score) in enumerate(zip(kbs.trials, kbs.scores), start=1)
            ]
        )
        trials["nan"] = trials.val_loss.apply(math.isnan)
        trials = trials.sort_values(["nan", "val_loss"]).drop(columns="nan")
        trials.to_csv(OUT_DIR / f"search_trials_{budget}a.csv", index=False)
        print(f"{budget} agents per model:\n{trials.to_string(index=False)}", flush=True)


if __name__ == "__main__":
    main()
