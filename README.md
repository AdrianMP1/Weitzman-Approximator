# Weitzman-Approximator

## Installation
### Prerequisities
* Conda (Miniconda or Anaconda).
* Python 3.10 or newer.

#### Activate a conda or venv environment
conda create -n myvenv python>=3.10
conda activate myvenv
conda install pip setuptools wheel

#### Install the package
From the project root (where pyproject.toml is located):
* pip install -e
This will allow changes to the source code to be reflected immediately.

## Project Structure
.
├── pyproject.toml
├── scripts/
├── configs/
├── data/
└── src/
    └── weitzman/
        ├── __init__.py
        ├── io/
        ├── utils/
        ├── metrics/
        ├── plotting/
        ├── reporting/
        ├── pipelines/
        └── algorithms/
