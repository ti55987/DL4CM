# Parameter estimation: RLF

Trains the GRU of Fig. 8B and Table A2, `training/models.get_model_for_parameter_estimation`, for every run of both.

| File                 | Contents                           |
|----------------------|------------------------------------|
| `rlf.py`             | trains and tests all 78 runs       |
| `search.py`          | the 18 hyperparameter searches     |
| `runs.csv`           | each run's seeds and configuration |
| `configurations.csv` | the best trial of each search      |

## Data

The test file is in `data/`. Simulate the two train files with `simulation/simulation.ipynb` (`model_name = "rlf"`, `N_AGENTS = 27000` and `9000`) and save them next to it under these names:

| File          | Path in `data/`                                |
|---------------|------------------------------------------------|
| Train, 27,000 | `rlf_6s3a/rand_500seqs_27000a_12b_78t_rlf.csv` |
| Train, 9,000  | `rlf_6s3a/rand_500seqs_9000a_12b_78t_rlf.csv`  |
| Test, 500     | `rlf_6s3a/rand500_st11_500a_12b_78t_rlf.csv`   |

## Run

From the repository root:

```
python training/local/parameter_estimation/rlf/rlf.py
python training/local/parameter_estimation/rlf/search.py
```

Both read `data/` and write to `results/parameter_estimation/rlf/`. `rlf.py` writes `recovery_per_agent.csv` (one row per test agent and Fig. 8B run, the ANN rows of the `ALL_RESULT_PATH` file of `eval/eval.ipynb`), `ablation_per_agent.csv` and `ablation_table.csv` (Table A2), `summary.csv` (one row per run and parameter), `{run_tag}_loss.csv`, `models/{run_tag}.keras` and `models/{run_tag}_label_scaler.pkl`; `search.py` writes `search_trials_{budget}a_{k}.csv` for each search and `search_best.csv`.

## Settings

- Features, Fig. 8B: the first 1 to 6 of `rewards`, `actions`, `prev_action`, `stim_iter`, `delay_since_last_stimuli`, `isswitch`, with the rows sorted by agent, block and stimulus.
- Features, Table A2: the six with `isswitch` replaced by `block_no` or `trials`, or removed; `stim_iter` replaced by `stimuli`, with the rows in trial order (`stim2stimuli_nosort`); or `prev_action` removed.
- Labels: `alpha`, `beta` (divided by 20), `phi`, `stickiness`, standardized on the training agents.
- Budgets, Fig. 8B: 1,000, 3,000 and 9,000 agents drawn from the 27,000-agent file with subsample seeds 27390, 24510 and 3634, and all 27,000 once.
- Budget, Table A2: all agents of the 9,000-agent file.
- Split: 10% of the agents for validation (`RandomState(42)`), both sets shuffled (`random_state=0`); the same 500 test agents.
- Training: up to 600 epochs, early stopping on the validation loss with patience 50, best weights restored.
- Init seed: 11; Table A2 also 52131 and 93939.
- Configurations: 27,000 agents use the 9,000-agent configuration of the same feature set; Table A2 uses the 9,000-agent configuration of all six features.
- Search: keras-tuner's `BayesianOptimization` (tuner seed 98560) over the search space of the workflow, 30 trials of up to 35 epochs with early stopping (patience 35), on the first 1,000, 3,000 or 9,000 agents of the 9,000-agent file; `configurations.csv` keeps each best trial to six significant digits, the values the models were trained with.

## Result

Fig. 8B: test RMSE averaged over the four parameters, the mean of the three seeds (one run at 27,000):

| Features     | 1,000  | 3,000  | 9,000  | 27,000 |
|--------------|--------|--------|--------|--------|
| r            | 0.1600 | 0.1510 | 0.1480 | 0.1425 |
| r-a          | 0.1640 | 0.1448 | 0.1318 | 0.1236 |
| r-a-p        | 0.1351 | 0.1305 | 0.1146 | 0.1065 |
| r-a-p-si     | 0.1389 | 0.1313 | 0.1124 | 0.1054 |
| r-a-p-si-d   | 0.1404 | 0.1298 | 0.1041 | 0.0906 |
| r-a-p-si-d-i | 0.1460 | 0.1301 | 0.1000 | 0.0904 |

Table A2: test RMSE averaged over the four parameters, mean ± SD over the three init seeds:

| Feature set           | RMSE            |
|-----------------------|-----------------|
| `orig`                | 0.1011 ± 0.0018 |
| `switch2block`        | 0.1037 ± 0.0023 |
| `switch2trial`        | 0.1072 ± 0.0017 |
| `noswitch`            | 0.1009 ± 0.0008 |
| `stim2stimuli_nosort` | 0.1140 ± 0.0020 |
| `noprev`              | 0.1221 ± 0.0011 |

The models were trained on one NVIDIA A10 GPU with TensorFlow 2.19.1 and Keras 3.12.4. There, `rlf.py` reproduces them exactly from the original train files, and all 78 runs take about 5 hours. Newly simulated train files hold other agents, so their numbers differ from the tables.
