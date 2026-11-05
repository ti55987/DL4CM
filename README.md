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

## Starting point

To go through the workflow of neural SBI for different use cases, running the following notebooks for different use cases:
- Model parameter estimation: `parameter_estimation_workflow.ipynb`
- Model comparison: `model_identification_workflow.ipynb`
- Trial-varying latent variable inference (LaseNet): `lasenet_workflow.ipynb`

To understand different usage of LaseNet (Pan et al 2025), please visit this [repo](https://github.com/ti55987/lasenet/tree/master)

## Contributing
If you'd like to contribute, or have any suggestions for these guidelines, open an issue on this GitHub repository.

All contributions welcome! All content in this repository is licensed under the MIT license.

## TODO

- simulation
  - rlf.py (Ti-Fen) ✅
  - wm.py (Ti-Fen) ✅
  - noise_rl.py (Nathan)
  - simulation.ipynb (Ti-Fen) ✅
- utils (Both)
- sim_environment.yml
- local_training_environment.yml (Nathan)
- training
  - local (Nathan)
    - parameter_estimation
      - rlf
      - noise_rl
    - model_comparison
      - rlf_wm

- benchmarks (Ti-Fen) ✅
  - mle
  - map
- eval (Ti-Fen) ✅
  - benchmark vs SBI 
  - permutation tests (added to workflow)

- parameter_estimation_workflow.ipynb (Ti-Fen) ✅
- model_comparison_workflow.ipynb (Ti-Fen) ✅
- lasenet_workflow.ipynb (Ti-Fen) ✅