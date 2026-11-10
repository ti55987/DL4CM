# Model comparison: RLF vs WM

Trains the classifier of Fig. 8C, `training/models.get_model_for_model_identification`, for every run of the figure.

| File                 | Contents                               |
|----------------------|----------------------------------------|
| `rlf_wm.py`          | trains and tests all 50 runs           |
| `search.py`          | the two searches behind `9k` and `27k` |
| `runs.csv`           | each run's init seed and configuration |
| `configurations.csv` | each configuration and its source      |

## Data

The test files are in `data/`. Simulate the two train files with `simulation/simulation.ipynb` (`model_name = "rlf"` and then `"wm"`, `N_AGENTS = 27000`, both on the same `all_seq`) and save them next to the test files under these names:

| File      | Path in `data/`                                |
|-----------|------------------------------------------------|
| RLF train | `rlf_6s3a/rand_500seqs_27000a_12b_78t_rlf.csv` |
| RLF test  | `rlf_6s3a/rand500_st11_500a_12b_78t_rlf.csv`   |
| WM train  | `wm_6s3a/rand500_st11_27000a_12b_78t_wm.csv`   |
| WM test   | `wm_6s3a/rand500_st11_500a_12b_78t_wm.csv`     |

## Run

From the repository root:

```
python training/local/model_comparison/rlf_wm/rlf_wm.py
python training/local/model_comparison/rlf_wm/search.py
```

Both read `data/` and write to `results/model_comparison/rlf_wm/`. `rlf_wm.py` writes `identification_per_agent.csv` (one row per test agent and run, the `ANN_RESULT_PATH` file of `eval/eval.ipynb`), `summary.csv` (one row per run), `{run_tag}_loss.csv` and `models/{run_tag}.keras`; `search.py` writes `search_trials_9000a.csv` and `search_trials_27000a.csv`.

## Settings

- Features: the first 1, 2, 3, 4 or 6 of `prev_action`, `rewards`, `actions`, `delay_since_last_stimuli`, `isswitch`, `stim_iter`, with the rows sorted by agent, block and stimulus; `stim_iter` is read from the simulated files.
- Budgets: 1,000, 3,000 and 9,000 agents per model, drawn with subsample seeds 5144, 1553 and 17544 (the same agent ids for both models), and all 27,000 agents once (seed 0).
- Split: 10% of each model's agents for validation (`random_state=42`), both sets shuffled (`random_state=0`); the same 500 test agents per model.
- Training: up to 600 epochs, early stopping on the validation loss with patience 50, best weights restored.
- Init seed: 123. A model whose best validation loss stayed at 0.690 or above (ln 2, chance) was retrained with a new random seed, up to three times with its budget's configuration and then, at 3,000 and 9,000 agents, twice with `9k_runner_up`; `runs.csv` keeps the attempt with the lowest validation loss and marks the 8 models that never trained as `dead`.
- Search: ten trials (sampling seed 33) of up to 35 epochs with early stopping (patience 35), on all six features and the first 9,000 or all 27,000 agents per model. The `9k` and `27k` configurations come from the 2026-08-24 search, which ran before the feature order above (it then started with `stim_iter`); rerun on this order at 9,000 agents, trial 1 (`9k_runner_up`) comes first and trial 5 (`9k`) stays at ln 2.

## Result

Test accuracy, the mean of the three seeds (one run at 27,000), as Fig. 8C plots it:

| Features    | 1,000 | 3,000 | 9,000 | 27,000 |
|-------------|-------|-------|-------|--------|
| p           | 0.501 | 0.506 | 0.507 | 0.500  |
| p-r         | 0.545 | 0.645 | 0.635 | 0.764  |
| p-r-a       | 0.550 | 0.787 | 0.766 | 0.905  |
| p-r-a-d     | 0.554 | 0.699 | 0.919 | 0.951  |
| p-r-a-d-i-s | 0.575 | 0.818 | 0.915 | 0.932  |

The models were trained on one NVIDIA A10 GPU with TensorFlow 2.19.1 and Keras 3.12.4. There, `rlf_wm.py` reproduces them exactly from the original train files, and all 50 runs take about 5 hours. Newly simulated train files hold other agents, so their numbers differ from the table.
