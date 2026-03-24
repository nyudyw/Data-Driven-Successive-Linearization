# Data-Driven Successive Linearization for Optimal Voltage Control

This repository contains the implementation and experiments for the paper:

Data-Driven Successive Linearization for Optimal Voltage Control

## Abstract

Power distribution systems are increasingly exposed to large voltage fluctuations driven by intermittent renewable generation and time varying loads (e.g., electric vehicles and storage). To address this challenge, a number of advanced controllers have been proposed for voltage regulation. However, these controllers typically rely on fixed linear approximations of voltage dynamics. As a result, the solutions may become infeasible when applied to the actual voltage behavior governed by nonlinear power flow equations, particularly under heavy power injection from distributed energy resources. This paper proposes a data-driven successive linearization approach for voltage control under nonlinear power flow constraints. By leveraging the fact that the deviation between the nonlinear power flow solution and its linearization is bounded by the distance from the operating point, we perform data-driven linearization around the most recent operating point. Convergence of the proposed method to a neighborhood of KKT points is established by exploiting the convexity of the objective function and structural properties of the nonlinear constraints. Case studies show that the proposed approach achieves fast convergence and adapts quickly to changes in net load.

## Repository Overview

The project includes:

- Power-flow models (linearized and nonlinear/DistFlow)
- Load profile preparation utilities
- Data-driven successive linearization voltage controller
- Other controllers (feedback optimization, convex relaxation, etc)
- Plotting and analysis tools for voltage and reactive power visualization
- Time-invariant and time-varying case studies

## Project Structure

```text
.
|-- README.md
|-- DataPred_n17_normalized.pkl
|-- configs/
|   |-- config_loader.py
|   |-- default_config.yaml
|   |-- distFlow_config.yaml
|   |-- distFlow33_config.yaml
|   |-- linDistFlow_config.yaml
|   `-- readme.md
|-- src/
|   |-- analysis/
|   |   `-- plot_tools.py
|   |-- cases/
|   |   |-- case33.py
|   |   `-- pypower/
|   |       |-- pypower_data_format.txt
|   |       |-- case33/
|   |       |   |-- bus.csv
|   |       |   |-- branch.csv
|   |       |   |-- gen.csv
|   |       |   |-- linear_k.pckl
|   |       |   |-- system_info.json
|   |       |   `-- README.txt
|   |       `-- case9/
|   |           |-- bus.csv
|   |           |-- branch.csv
|   |           |-- gen.csv
|   |           |-- system_info.json
|   |           `-- README.txt
|   |-- controllers/
|   |   `-- voltvar_controllers.py
|   |-- models/
|   |   `-- PF_models.py
|   `-- utils/
|       `-- case_loader.py
|-- convrelax.py
|-- dataloaders.py
|-- pfmodels.py
|-- utilfuncs.py
|-- plot_util.py
|-- timeinvariant_main.ipynb
`-- timevarying_main.ipynb
```

## File Guide


### notebooks

- timeinvariant_main.ipynb: notebook for time-invariant experiments.
- timevarying_main.ipynb: notebook for time-varying experiments.


### scripts

- convrelax.py: script for convex-relaxation-based experiments.
- dataloaders.py: build and transform time-series net-load inputs for control experiments.
- pfmodels.py: power-flow model implementations used by experiments.
- plot_util.py: plotting and reporting utilities for figures and diagnostics.
- utilfuncs.py: helper functions for gradients, costs, and voltage-related post-processing.

### configs

- config_loader.py: YAML configuration loader.
- default_config.yaml: baseline experiment defaults.
- linDistFlow_config.yaml: configuration for linearized DistFlow experiments.
- distFlow_config.yaml: configuration for DistFlow experiments.
- distFlow33_config.yaml: IEEE 33-bus DistFlow configuration.


### src/cases

- case33.py: case construction/helper for the IEEE 33-bus benchmark.

### src/controllers

- voltvar_controllers.py: distributed/centralized Volt/Var controller implementations.

### src/models

- PF_models.py: PFModel, LinDistFlow, and DistFlow classes.

### src/utils

- case_loader.py: case-loading helpers.

## Environment Setup

### 1) Clone repository

```bash
git clone https://github.com/<your-username>/Data-Driven-Successive-Lineraization.git
cd Data-Driven-Successive-Lineraization
```

### 2) Create Python environment

```bash
conda create -n ddsl python=3.11 -y
conda activate ddsl
pip install numpy scipy matplotlib pyyaml cvxpy
```

If you use MOSEK for CVXPY optimization, install and activate a valid MOSEK license separately.

## Running Experiments

### Notebook workflows

- timeinvariant_main.ipynb for time-invariant load studies.
- timevarying_main.ipynb for time-varying load studies.

## Citation

If you use this code in academic work, please cite the paper:

- Data-Driven Successive Linearization for Optimal Voltage Control
- arXiv: https://arxiv.org/abs/2603.10138

