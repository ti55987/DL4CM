# Parameter estimation: noisy RL

Trains the GRU of `training/models.get_model_for_parameter_estimation` on the noisy RL agents of the appendix: the models of Table B2, the same models at 9,000 agents, and the tuned models of the 12 hyperparameter searches.

| File                 | Contents                                |
|----------------------|-----------------------------------------|
| `noise_rl.py`        | trains and tests all 72 runs            |
| `search.py`          | the 12 searches behind the tuned models |
| `runs.csv`           | each run's init seed and configuration  |
| `configurations.csv` | each configuration and its source       |

## Data

The three files are simulated with `simulation/noise_rl.py`. From the repository root, in Python:

```python
import os

import numpy as np

from simulation.noise_rl import (
    generate_noise_rl_parameters,
    simulate_noisy_rl_agents,
)

os.makedirs("data/noisyrl", exist_ok=True)
name = (
    "data/noisyrl/sim_data_noisy_rl_exp3_nAgents{}_beta35"
    "_alphaMax0.8_repBias1_weber1_obsNoise1_cfLearn1.csv.gz"
)

np.random.seed(20260916)
params = generate_noise_rl_parameters(3500)
agents = simulate_noisy_rl_agents(params)
train_ids = np.random.choice(3500, 3000, replace=False)
is_train = agents.subj_code.isin(train_ids)
agents[is_train].to_csv(name.format(3000), index=False)
agents[~is_train].to_csv(name.format(500), index=False)

np.random.seed(20260917)
params = generate_noise_rl_parameters(9000)
agents = simulate_noisy_rl_agents(params)
agents.to_csv(name.format(9000), index=False)
```

The first seed gives the 3,000 training and 500 test agents, the second the 9,000 training agents; simulating them takes about 1.5 hours.

## Run

From the repository root:

```
python training/local/parameter_estimation/noise_rl/noise_rl.py
python training/local/parameter_estimation/noise_rl/search.py
```

Both read `data/` and write to `results/parameter_estimation/noise_rl/`. `noise_rl.py` writes `recovery_per_agent.csv` (one row per test agent and run, the `ALL_RESULT_PATH` file of the noisy RL section of `eval/eval.ipynb`), `recovery_summary.csv` (one row per run and parameter), `feature_importance.csv` (one row per run and feature), `{run_tag}_loss.csv`, `models/{run_tag}.keras` and `models/{run_tag}_label_scaler.pkl`; `search.py` writes `search_trials_{budget}a_{feature_set}.csv` for each search.

## Settings

- Agents: experiment 3 of Findling et al. (2019), 16 blocks of 56 trials; `beta` 35, `alpha_c` and `alpha_u` U(0.05, 0.8), `epsilon` U(0.01, 1), `rep_bias` U(-0.25, 0.5).
- Labels: `alpha_c`, `alpha_u`, `epsilon` and `xi` (the simulated `rep_bias`, κ in Table B2), standardized with the mean and standard deviation of the training agents.
- Features: `base4` is `actions`, `prev_action`, `chosen_vs_unchosen_reward_bin` and `reward1_bin`; `five`, `six`, `seven`, `no_optimal` and `all9` add `reward0_bin`, `chosen_reward_bin`, `unchosen_reward_bin`, `block_start` and `optimal_choice` one at a time. Each payoff is cut into five equal bins on [0, 1], the chosen minus unchosen payoff on [-1, 1]; `prev_action` is -1 on an agent's first trial only; the rows are sorted by agent, block and trial.
- Split: 10% of the training agents validate, drawn with the seed that simulated them (20260916 or 20260917); the same 500 test agents for every run.
- Training: up to 600 epochs, early stopping on the validation loss with patience 50, best weights restored; init seeds 11, 15355 and 79382.
- Search: ten trials (sampling seed 11) of up to 35 epochs with early stopping (patience 35), after one epoch of the `default` configuration at the search's `search_seed`, the init seed whose `default` model of that budget and feature set has the lowest validation loss; a tuned model uses the winner of its search, rounded to ten decimals.
- Feature importance: `utils/feature_importance_utils.permutation_importance_global_sequence` on the test agents, labels standardized with the training scaler, `within_sample`, 3 repeats, batch 64, `random_state=123`.

## Result

RMSE on the 500 test agents, the mean of the three init seeds, `default` configuration. 3,000 agents (Table B2):

| Feature set | alpha_c | alpha_u | epsilon | xi    | overall |
|-------------|---------|---------|---------|-------|---------|
| base4       | 0.168   | 0.145   | 0.128   | 0.079 | 0.130   |
| five        | 0.178   | 0.160   | 0.119   | 0.077 | 0.133   |
| six         | 0.169   | 0.159   | 0.120   | 0.076 | 0.131   |
| seven       | 0.190   | 0.191   | 0.117   | 0.077 | 0.144   |
| no_optimal  | 0.171   | 0.183   | 0.120   | 0.078 | 0.138   |
| all9        | 0.171   | 0.167   | 0.120   | 0.081 | 0.135   |

9,000 agents:

| Feature set | alpha_c | alpha_u | epsilon | xi    | overall |
|-------------|---------|---------|---------|-------|---------|
| base4       | 0.121   | 0.120   | 0.103   | 0.070 | 0.104   |
| five        | 0.119   | 0.117   | 0.098   | 0.069 | 0.101   |
| six         | 0.115   | 0.116   | 0.101   | 0.069 | 0.100   |
| seven       | 0.120   | 0.115   | 0.101   | 0.072 | 0.102   |
| no_optimal  | 0.114   | 0.113   | 0.098   | 0.064 | 0.097   |
| all9        | 0.115   | 0.115   | 0.095   | 0.068 | 0.098   |

Overall RMSE of the `default` and the tuned models:

| Feature set | 3,000 | 3,000 tuned | 9,000 | 9,000 tuned |
|-------------|-------|-------------|-------|-------------|
| base4       | 0.130 | 0.159       | 0.104 | 0.110       |
| five        | 0.133 | 0.151       | 0.101 | 0.104       |
| six         | 0.131 | 0.146       | 0.100 | 0.105       |
| seven       | 0.144 | 0.145       | 0.102 | 0.106       |
| no_optimal  | 0.138 | 0.153       | 0.097 | 0.102       |
| all9        | 0.135 | 0.142       | 0.098 | 0.097       |

Feature importance of the `all9` models (`default`), the rise of the test loss, mean of the three init seeds:

| Feature                       | 3,000 | 9,000 |
|-------------------------------|-------|-------|
| actions                       | 1.029 | 1.680 |
| prev_action                   | 0.656 | 0.777 |
| reward1_bin                   | 0.193 | 0.166 |
| chosen_vs_unchosen_reward_bin | 0.141 | 0.080 |
| reward0_bin                   | 0.128 | 0.134 |
| chosen_reward_bin             | 0.107 | 0.088 |
| unchosen_reward_bin           | 0.038 | 0.040 |
| block_start                   | 0.024 | 0.065 |
| optimal_choice                | 0.007 | 0.008 |

The models were trained on one NVIDIA A10 GPU with TensorFlow 2.19.1 and Keras 3.12.4. There, `noise_rl.py` reproduces them exactly, and all 72 runs take about 4 hours.
