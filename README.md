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

## RL-CDVAE Workflow

### CDVAE Pretraining

Our custom Predictor module is enabled by default. Therefore, before training CDVAE, ensure that the target property is correctly configured in `conf/predictor/default.yaml`.

To train a CDVAE, run the following command:

```
python cdvae/run.py data=mp_20 expname=mp_20
```

To use other datasets, use `data=con_mp_20` and `data=oqmd_20` instead. The corresponding configuration files are located in `conf/data/`.

After training, model checkpoints can be found in `$HYDRA_JOBS/singlerun/YYYY-MM-DD/expname`.

### Predictor Evaluation

After training CDVAE, you can optionally evaluate the performance of our custom Predictor module:

```bash
python scripts/predictor_evaluate.py --model_path MODEL_PATH --prop PROPERTY_NAME
```

- `MODEL_PATH`: The path to the trained model.
- `PROPERTY_NAME`: The target property to be evaluated. Currently, `band_gap` and `formation_energy` are supported.

The evaluation results are saved as PNG files in the `MODEL_PATH` directory.

Users can modify `scripts/predictor_evaluate.py` to support additional properties as needed.

### RL Modulation

Perform reinforcement learning training to modulation the sampling distribution in the latent space:

```bash
python scripts/train_ppo.py --model_path MODEL_PATH --surrogate SURROGATE_MODEL --data DATA --task TASK --reward_type REWARD_TYPE --target TARGET --epochs EPOCHS
```

- `MODEL_PATH`: The path to the trained CDVAE model.
- `SURROGATE_MODEL`: The surrogate model used for property prediction. Currently, [ComFormer](https://github.com/divelab/AIRS/tree/main/OpenMat/ComFormer) and [CGCNN](https://github.com/txie-93/cgcnn) are supported.
- `DATA`: The dataset to be used. Supported options are `mp_20`, `con_mp_20`, and `oqmd_20`.
- `TASK`: The target property to be optimized. Supported options are `band_gap` and `formation_energy`.
- `REWARD_TYPE`: The type of optimization objective. Supported options are:
  - `max_prop`: Maximize the target property.
  - `min_prop`: Minimize the target property.
  - `single_target`: Optimize the target property toward a specified value.
  - `interval_target`: Optimize the target property to fall within a specified interval.
- `TARGET`: The target value or interval for optimization. This argument is only used when `REWARD_TYPE` is set to `single_target` or `interval_target`.
  - For `single_target`: e.g., `3` or `-3`.
  - For `interval_target`: e.g., `3~5` or `-2~1`.
- `EPOCHS`: The number of training epochs.

The results are saved in the `{TASK}_{TARGET}` subdirectory under `MODEL_PATH`.

Users can modify `scripts/train_ppo.py` to accommodate their specific optimization tasks and define custom reward functions in `ppo/reward.py` as needed.

### Generation

Once reinforcement learning training has converged, crystal structures can be generated using the following command:

```bash
python scripts/evaluate.py --model_path MODEL_PATH --tasks gen --ppo_task PPO_TASK --ppo --filter_z
```

- `MODEL_PATH`: The path to the trained CDVAE model.
- `PPO_TASK`: The directory containing the trained reinforcement learning agent, located within the `MODEL_PATH` directory.
- `--ppo`: Enable the trained reinforcement learning agent to optimize latent variables during generation.
- `--filter_z`: Enable our custom Predictor module to filter latent variables during generation.

The generated crystal structures are saved as `.pt` files in the `MODEL_PATH` directory.

These files are PyTorch serialized files containing multiple tensors that describe the structures of `M` materials batched together. Each material may contain a different number of atoms, with `N` denoting the total number of atoms across all materials. `num_evals` denotes the number of Langevin dynamics runs performed for each material.

- `frac_coords`: Fractional coordinates of each atom, shape `(num_evals, N, 3)`.
- `atom_types`: Atomic number of each atom, shape `(num_evals, N)`.
- `lengths`: Lattice lengths, shape `(num_evals, M, 3)`.
- `angles`: Lattice angles, shape `(num_evals, M, 3)`.
- `num_atoms`: Number of atoms in each material, shape `(num_evals, M)`.

We recommend specifying the additional `--label` argument during generation to distinguish the output files from different generation tasks.

### 5. Generation Metrics Evaluation

To compute evaluation metrics, run the following command:

```
python scripts/compute_metrics.py --root_path MODEL_PATH --tasks gen
```

`MODEL_PATH` will be the path to the trained model. All evaluation metrics will be saved in `eval_metrics.json`.

## Acknowledgements

This project builds upon the [Crystal Diffusion Variational Autoencoder (CDVAE)](https://github.com/txie-93/cdvae) framework.

A small portion of the reinforcement learning implementation is adapted from [MolRL](https://github.com/Cellarity/MolRL), and the adapted code remains subject to its original license.

We employ [CGCNN](https://github.com/txie-93/cgcnn) and [ComFormer](https://github.com/divelab/AIRS/tree/main/OpenMat/ComFormer) as surrogate models for crystal property prediction.

We sincerely thank the authors and contributors of these projects for making their code publicly available and facilitating further research in this field.

## Citation



## Contact

Please leave an issue or reach out to the author(czhang26@mails.jlu.edu.cn) if you have any questions.
