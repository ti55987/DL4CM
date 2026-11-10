"""Hyperparameter search for the RLF GRU: the 18 searches behind configurations.csv.

keras-tuner's BayesianOptimization (tuner seed 98560), one search per budget (the first 1,000, 3,000 and 9,000 agents
of the 9,000-agent file) and cumulative feature set: 30 trials of up to 35 epochs (early stopping, patience 35),
scored by their best validation loss. configurations.csv holds each search's best trial to six significant digits, the
values the models were trained with. The script reads data/ and writes search_trials_{budget}a_{k}.csv (one row per
trial, best first) and search_best.csv (the best trial of each search) to results/parameter_estimation/rlf/.

    python training/local/parameter_estimation/rlf/search.py
"""

import argparse
import os

import keras_tuner as kt
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import EarlyStopping

from rlf import (
    CUMULATIVE_SETS,
    DATA_DIR,
    DATA_FILES,
    OUT_DIR,
    SORTED_COL,
    get_iteration_features,
    get_labels,
    read_data,
)
from training.models import get_model_for_parameter_estimation
from utils.feature_utils import extract_features_blockless, normalize_train_labels, normalize_val_labels

BUDGETS = [1000, 3000, 9000]  # the first agents of the 9,000-agent file
TUNER_SEED = 98560
MAX_TRIALS = 30
EPOCHS = 35
PATIENCE = 35


def build_model_factory(feature_dim: int, output_dims: int):
    """The search space of parameter_estimation_workflow.ipynb, written for keras-tuner."""

    def build_model(hp):
        params = {
            "feature_dim": feature_dim,
            "output_dims": output_dims,
            "units": hp.Choice("units", [96, 128, 512]),
            "learning_rate": hp.Float("learning_rate", 3e-4, 3e-3, sampling="log"),
            "dropout": hp.Float("dropout", 0.015, 0.25),
            "dropout1": hp.Float("dropout1", 0.01, 0.2),
            "dropout2": hp.Float("dropout2", 0.01, 0.2),
        }
        hp.Choice("batch_size", [128, 256, 512])
        return get_model_for_parameter_estimation(params)

    return build_model


class BatchSizeBayesianOptimization(kt.BayesianOptimization):
    """Fits each trial with its sampled batch_size, which keras-tuner would otherwise leave unused."""

    def run_trial(self, trial, *args, **kwargs):
        batch_size = trial.hyperparameters.values.get("batch_size")
        if batch_size is not None:
            kwargs["batch_size"] = batch_size
        return super().run_trial(trial, *args, **kwargs)


def main():
    # No options: the 18 searches run in turn. --help prints the docstring.
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    data = read_data(DATA_DIR / DATA_FILES["9000a"])

    best = []
    for budget in BUDGETS:
        train_df = data[data.agentid.isin(np.sort(data.agentid.unique())[:budget])]
        val_agent_ids = np.random.RandomState(42).choice(
            train_df.agentid.unique(), int(train_df.agentid.nunique() * 0.1), replace=False
        )
        normalized_train_labels, name_to_scaler = normalize_train_labels(
            get_labels(train_df[~train_df.agentid.isin(val_agent_ids)])
        )
        normalized_val_labels = normalize_val_labels(
            get_labels(train_df[train_df.agentid.isin(val_agent_ids)]), name_to_scaler
        )
        td_iter_order = get_iteration_features(train_df).sort_values(SORTED_COL)
        training_data = td_iter_order[~td_iter_order.agentid.isin(val_agent_ids)]
        val_data = td_iter_order[td_iter_order.agentid.isin(val_agent_ids)]

        for k, (feature_set, feature_list) in enumerate(CUMULATIVE_SETS.items(), 1):
            train_features, train_labels = shuffle(
                extract_features_blockless(training_data, feature_list, []), normalized_train_labels, random_state=0
            )
            val_features, val_labels = shuffle(
                extract_features_blockless(val_data, feature_list, []), normalized_val_labels, random_state=0
            )
            tuner = BatchSizeBayesianOptimization(
                build_model_factory(train_features.shape[-1], train_labels.shape[-1]),
                objective="val_loss",
                max_trials=MAX_TRIALS,
                seed=TUNER_SEED,
                max_consecutive_failed_trials=MAX_TRIALS,
                directory=os.path.join(OUT_DIR, "keras_tuner"),
                project_name=f"rlf_{budget}a_{k}",
                overwrite=True,
            )
            tuner.search(
                train_features,
                train_labels,
                validation_data=(val_features, val_labels),
                epochs=EPOCHS,
                callbacks=[EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)],
                verbose=0,
            )
            trials = pd.DataFrame(
                [
                    {"trial": t.trial_id, "val_loss": t.score, **t.hyperparameters.values}
                    for t in tuner.oracle.trials.values()
                    if t.score is not None
                ]
            ).sort_values("val_loss")
            trials.to_csv(os.path.join(OUT_DIR, f"search_trials_{budget}a_{k}.csv"), index=False)
            best.append(
                {
                    "configuration": f"{budget}a_{k}",
                    "budget": budget,
                    "feature_set": feature_set,
                    **trials.iloc[0].to_dict(),
                }
            )
            print(pd.DataFrame(best[-1:]).to_string(index=False), flush=True)
            pd.DataFrame(best).to_csv(os.path.join(OUT_DIR, "search_best.csv"), index=False)


if __name__ == "__main__":
    main()
