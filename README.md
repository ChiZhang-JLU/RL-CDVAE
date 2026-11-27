# RL-CDVAE
A reinforcement learning framework for conditional crystal structure generation based on CDVAE.

This project builds upon the [Crystal Diffusion Variational Autoencoder (CDVAE)](https://github.com/txie-93/cdvae) proposed by [Xie et al. (ICLR 2022)](https://arxiv.org/abs/2110.06197) and integrates reinforcement learning to guide crystal structure generation toward desired target properties.


## Contents

- [Setup](#setup)
- [Example workflow on MP-20](#example-workflow-on-mp-20)
  1. [Train the CDVAE](#1-train-the-cdvae)
  2. [Evaluate the CDVAE predictor](#2-evaluate-the-cdvae-predictor)
  3. [Train the PPO sampling policy](#3-train-the-ppo-sampling-policy)
  4. [Generate structures and save a test reference](#4-generate-structures-and-save-a-test-reference)
  5. [Compute generation metrics](#5-compute-generation-metrics)
- [Outputs](#outputs)
- [Repository layout](#repository-layout)

## Setup

Run the commands from the repository root in a Linux Bash shell. The training configuration requests one CUDA GPU by default. The shell scripts are examples for a Slurm cluster. The Python commands below can also run directly on a GPU machine.

Use a Python environment with these dependencies.

- **Training and sampling** use PyTorch, PyTorch Lightning, PyTorch Geometric, `torch-scatter`, `torch-sparse`, Hydra, OmegaConf, `python-dotenv`, SymPy, and `wandb`.
- **Crystal data and evaluation** use NumPy, pandas, SciPy, scikit-learn, pymatgen, NetworkX, `p_tqdm`, `tqdm`, Matplotlib, SMACT, and matminer.
- **PPO and ComFormer** also use `jarvis-tools`, `e3nn`, `pandarallel`, and the Pydantic v1 API. The current ComFormer code imports Python's legacy `imp` module.

The repository does not pin dependency versions. Match the PyTorch Geometric extensions to your PyTorch and CUDA builds, and use a PyTorch Lightning release that supports the `Trainer` arguments in [`cdvae/run.py`](cdvae/run.py). The model loaders also expect a PyTorch environment that can load the serialized scaler objects produced during training.

From the repository root, make the local modules available on the Python path.

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

Use [`.env.template`](.env.template) as a starting point for a local `.env`. Set the paths to match your checkout, then create the output and log directories.

```bash
export PROJECT_ROOT="/absolute/path/to/rl-cdvae"
export HYDRA_JOBS="/absolute/path/to/rl-cdvae/hydra"
export WABDB_DIR="/absolute/path/to/rl-cdvae/wandb"
```

`WABDB_DIR` is the variable spelling used by the Hydra configuration for the W&B log directory. Before running the predictor evaluation, set `font_path` in [`scripts/predictor_evaluate.py`](scripts/predictor_evaluate.py) to a font file available on your machine. If you use Slurm, update the partition, environment activation, library paths, and experiment arguments in the launchers to match your cluster and run.

## Example workflow on MP-20

This example trains a CDVAE with a band-gap prediction head and uses the ComFormer surrogate to guide generation toward larger predicted band gaps. Use one CDVAE model directory throughout and the same label for both sampling and evaluation.

### 1. Train the CDVAE

Check that [`data/mp_20`](data/mp_20) contains `train.csv`, `val.csv`, and `test.csv`. Each split contains crystal structures as CIF text in the `cif` column, sample identifiers in `material_id`, and property labels including `band_gap`.

```bash
python cdvae/run.py data=mp_20 expname=mp_20
```

The training run writes `hparams.yaml`, a `.ckpt` checkpoint, `lattice_scaler.pt`, and `prop_scaler.pt` under `$HYDRA_JOBS/singlerun/<date>/mp_20/`. Set `MODEL_DIR` to that run's absolute path.

```bash
MODEL_DIR="/absolute/path/to/hydra/singlerun/YYYY-MM-DD/mp_20"
LABEL="ppo_bg_max"
```

### 2. Evaluate the CDVAE predictor

```bash
python scripts/predictor_evaluate.py \
  --model_path "$MODEL_DIR" \
  --prop band_gap
```

The script evaluates the CDVAE's latent-space prediction head on the configured test split. It saves `band_gap_scatter.png` with the MAE, MSE, RMSE, and R² annotated on the plot.

### 3. Train the PPO sampling policy

The MP-20 band-gap example reads a `best*.pt` checkpoint and `scaler.pt` from [`comformer/mp_20/band_gap`](comformer/mp_20/band_gap). The example keeps PPO's W&B logs local by running in offline mode.

```bash
WANDB_MODE=offline python scripts/train_ppo.py \
  --model_path "$MODEL_DIR" \
  --predictor comformer --data mp_20 --task band_gap \
  --reward_type max_prop --epochs 300 --batch_size 32
```

The policy and its training logs go into `$MODEL_DIR/band_gap_max/`. The generator loads `best_policy_net.pt` from this directory. Other available reward modes are `min_prop`, `single_target`, and `interval_target`. For fixed and interval targets, use `--target` with a numeric value or a `lower~upper` range. Match the surrogate checkpoint to the dataset and property you select.

### 4. Generate structures and save a test reference

```bash
python scripts/evaluate.py \
  --model_path "$MODEL_DIR" \
  --tasks recon gen --label "$LABEL" \
  --ppo --ppo_task band_gap_max
```

`gen` samples new crystal structures with the learned policy. `recon` reconstructs the configured test set and saves the input structures used as the reference for evaluation. To generate a baseline from the same CDVAE, omit `--ppo` and `--ppo_task` and choose a new label.

### 5. Compute generation metrics

```bash
python scripts/compute_metrics.py \
  --root_path "$MODEL_DIR" \
  --tasks gen --label "$LABEL"
```

The evaluator reads `eval_gen_<label>.pt` and the reference structures from `eval_recon_<label>.pt`. Both files need the same label. Generation metrics require at least 500 valid generated crystals in the first decoding repeat. The JSON report covers composition and structure validity, fingerprint diversity, density and element-count distributions, and coverage of the test set.

## Outputs

| Path relative to `MODEL_DIR` | Contents |
| --- | --- |
| `hparams.yaml`, `*.ckpt`, `lattice_scaler.pt`, `prop_scaler.pt` | CDVAE configuration, weights, and scalers |
| `band_gap_scatter.png` | Test-set predictions and regression metrics |
| `band_gap_max/best_policy_net.pt` | PPO policy used for generation |
| `band_gap_max/train_metrics.csv`, `band_gap_max/reward_curve.png` | PPO training statistics and reward history |
| `eval_gen_ppo_bg_max.pt`, `eval_recon_ppo_bg_max.pt` | Generated and reconstructed structures, with test inputs in the reconstruction file |
| `eval_metrics_ppo_bg_max.json` | Summary statistics for the generated structures |

The structure files store fractional coordinates (`frac_coords`), atomic numbers (`atom_types`), atoms per crystal (`num_atoms`), lattice lengths (`lengths`), and lattice angles (`angles`) as tensors.

## Repository layout

| Location | Purpose |
| --- | --- |
| [`cdvae/`](cdvae), [`conf/`](conf) | Crystal model, data loading, and Hydra configuration |
| [`ppo/`](ppo), [`comformer/`](comformer), [`cgcnn/`](cgcnn) | Policy training and crystal property surrogates |
| [`scripts/`](scripts) | Predictor evaluation, PPO training, sampling, and metrics |
| [`data/`](data) | Dataset splits and dataset notes |
| [`train.sh`](train.sh), [`predictor_evaluate.sh`](predictor_evaluate.sh), [`ppo.sh`](ppo.sh), [`gen.sh`](gen.sh), [`metrics.sh`](metrics.sh) | Example Slurm launchers in workflow order |

See the [dataset notes](data/README.md) for upstream benchmark references and the [MP-20 notes](data/mp_20/README.md) for dataset background.
