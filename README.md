# RL-CDVAE
A reinforcement learning framework for conditional crystal structure generation based on CDVAE.

This project builds upon the [Crystal Diffusion Variational Autoencoder (CDVAE)](https://github.com/txie-93/cdvae) proposed by [Xie et al. (ICLR 2022)](https://arxiv.org/abs/2110.06197) and integrates reinforcement learning to guide crystal structure generation toward desired target properties.


## Contents

- [Installation](#installation)
- [Datasets](#datasets)
- [RL-CDVAE Workflow](rl-cdvae-workflow)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

## Installation

We recommend using Conda for environment management, which can be downloaded from the [official Conda website](https://docs.conda.io/).

### GPU machines

Create the environment using the `env.yml` file:
```bash
conda env create -f env.yml
```
Activate the conda environment with `conda activate cdvae`.

Install this package with `pip install -e .`.

### Faster installation

An alternative way to install the required packages more quickly is:
```bash
conda env create -f env_sub.yml
conda activate cdvae
conda install ipywidgets jupyterlab matplotlib pylint
conda install -c conda-forge matminer=0.7.3 nglview pymatgen=2020.12.31
# Downgrade to fix a known issue with pytorch
python3 -m pip install setuptools==59.5.0
pip install -e .
```

### Setting up environment variables

Make a copy of the `.env.template` file and rename it to `.env`. Modify the following environment variables in `.env`.

- `PROJECT_ROOT`: path to the folder that contains this repo
- `HYDRA_JOBS`: path to a folder to store hydra outputs
- `WABDB`: path to a folder to store wabdb outputs

## Datasets

All datasets are directly available on `data/` with train/val/test splits. If you use these datasets, please consider to cite the original papers from which we curate these datasets.
