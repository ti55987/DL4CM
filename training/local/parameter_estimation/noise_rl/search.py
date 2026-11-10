"""Hyperparameter search for the noisy RL GRU: the 12 searches behind the tuned configurations of configurations.csv.

KerasBayesianSearch, as in the workflow notebooks: for each budget and feature set, ten trials (sampling seed 11) of
up to 35 epochs (early stopping, patience 35), scored by their best validation loss. Each search runs at its
`search_seed` in configurations.csv, the init seed whose `default` model of that budget and feature set has the lowest
validation loss. The script reads data/ and writes to results/parameter_estimation/noise_rl/.

    python training/local/parameter_estimation/noise_rl/search.py

Writes search_trials_{budget}a_{feature_set}.csv for each search, one row per trial, best first (trials whose loss
went NaN last).
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

from noise_rl import DATA_DIR, FEATURE_LIST, FEATURE_SETS, HERE, OUT_DIR, PARAMETERS, TRAIN_FILES, prepare_dataset
from training.models import get_model_for_parameter_estimation
from utils.model_utils import set_seed

SAMPLING_SEED = 11
HP_KEYS = ["units", "dropout", "dropout1", "dropout2", "learning_rate", "batch_size"]


def main():
    # No options: the 12 searches run in turn. --help prints the docstring.
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    configurations = pd.read_csv(HERE / "configurations.csv").set_index("configuration")
    default = configurations.loc["default"]

    for budget in TRAIN_FILES:
        dataset = prepare_dataset(DATA_DIR, budget)
        for feature_set, features in FEATURE_SETS.items():
            name = f"{budget}a_{feature_set}"
            print(f"search {name}", flush=True)
            columns = [FEATURE_LIST.index(f) for f in features]
            X_train, y_train = shuffle(dataset["X_train"][:, :, columns], dataset["y_train"], random_state=0)
            X_val, y_val = shuffle(dataset["X_val"][:, :, columns], dataset["y_val"], random_state=0)

            # The searches of configurations.csv ran right after one epoch of the default configuration at the same
            # seed. The same epoch here leaves the random state where those trials found it, so the trials reproduce.
            tf.keras.backend.clear_session()
            set_seed(int(configurations.loc[name, "search_seed"]))
            model = get_model_for_parameter_estimation(
                {
                    "feature_dim": len(features),
                    "output_dims": len(PARAMETERS),
                    "units": int(default.units),
                    "dropout": float(default.dropout),
                    "dropout1": float(default.dropout1),
                    "dropout2": float(default.dropout2),
                    "learning_rate": float(default.learning_rate),
                }
            )
            model.fit(
                X_train,
                y_train,
                epochs=1,
                batch_size=int(default.batch_size),
                callbacks=[EarlyStopping(monitor="val_loss", patience=1, restore_best_weights=True)],
                validation_data=(X_val, y_val),
                verbose=2,
            )

            param_grid = {
                "feature_dim": len(features),
                "output_dims": len(PARAMETERS),
                "units": hp.choice("units", [96, 128, 512]),
                "learning_rate": hp.loguniform("learning_rate", np.log(3e-4), np.log(3e-3)),
                "dropout": hp.uniform("dropout", 0.015, 0.25),
                "dropout1": hp.uniform("dropout1", 0.01, 0.2),
                "dropout2": hp.uniform("dropout2", 0.01, 0.2),
                "epochs": 35,
                "batch_size": hp.choice("batch_size", [128, 256, 512]),
            }
            kbs = KerasBayesianSearch(
                get_model_for_parameter_estimation,
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
            trials.to_csv(OUT_DIR / f"search_trials_{name}.csv", index=False)
            print(trials.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
