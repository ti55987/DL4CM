# Neural Simulation-Based Inference for Computational Cognitive Models: A Tutorial

This repository is the official implementation of [Neural Simulation-Based Inference (SBI) for Computational Cognitive Models: A Tutorial](TODO)

## Requirements

To install requirements, first create a conda environment from the environment_kera3 yaml file by running

```setup
conda env create -f environment_kera3.yml
```
Activate this environment by running
```
conda activate dl4rl-k3
```

## Getting Started

To walk through the neural SBI workflow, check out the following notebooks for your use case:
- Model parameter estimation: `parameter_estimation_workflow.ipynb`
- Model comparison: `model_identification_workflow.ipynb`
- Trial-varying latent variable inference (LaseNet): `lasenet_workflow.ipynb`
  - For other uses of LaseNet [(Pan et al 2025)](https://link.springer.com/article/10.3758/s13428-025-02794-0), please visit this [repo](https://github.com/ti55987/lasenet/tree/master)

## Code structure
- `simulation`: code for simulating the models and tasks in the tutorial (i.e., RLF, WM, and Noisy RL models).
- `data`: 500 held-out test agents for the RLF and WM models, generated with the simulation code above.
- `training`: neural network models for each use case.
- `pretrained`: pretrained Keras models used in the demonstrations (workflow notebook).
- `benchmarks`: likelihood-based estimators (MLE and MAP) used to benchmark neural SBI in the paper.
- `eval`: code for the quantitative and qualitative evaluation figures in the paper.
- `utils`: helper functions for training, evaluation, feature extraction...etc

## Contributing
If you'd like to contribute, or have any suggestions for these guidelines, open an issue on this GitHub repository.

All contributions welcome! All content in this repository is licensed under the MIT license.

## TODO

- simulation ✅
- utils (Both)
- enviornment_kera3.yml (Both)
- training
  - models (Ti-Fen) ✅
  - local (Nathan)
    - parameter_estimation
      - rlf
      - noise_rl
    - model_comparison
      - rlf_wm

- benchmarks (Ti-Fen) ✅
- eval (Ti-Fen) ✅
- parameter_estimation_workflow.ipynb (Ti-Fen) ✅
- model_comparison_workflow.ipynb (Ti-Fen) ✅
- lasenet_workflow.ipynb (Ti-Fen) ✅