# Federated Learning / Prototype Federated Learning (CIFAR-10)

Research-oriented codebase for running CIFAR-10 experiments under **non-IID** data partitions, including:
- **Standalone baseline** training per client (no aggregation) for sanity checks
- **Federated Learning (FL)** simulations using Flower
- **Prototype Federated Learning (PFL)**-style training where clients exchange **prototypes** (not full model weights)
- A modular **security** layer to plug in attacks/defenses and log metrics per run

## What’s inside

- **Entry points**
  - `download_data.py`: downloads CIFAR-10 into `data/`
  - `run_baseline.py`: independent local training per client + evaluation variants
  - `run_fl.py`: Flower-based FL simulation (Ray simulation by default)
  - `run_pfl.py`: Flower-based PFL simulation (prototype exchange)
- **Core package**: `src/`
  - `src/data_utils.py`: CIFAR-10 loading + Dirichlet split helpers
  - `src/models.py`: ResNet-18 for CIFAR-10
  - `src/train_utils*.py`: training/eval utilities
  - `src/aggregation.py`: prototype aggregation strategy
  - `src/security/`: attacks, defenses, metrics, and run-level logging
- **Notebooks**: `colab/` (ready-to-run experiment notebooks)
- **Generated artifacts**
  - `data/`: downloaded CIFAR-10
  - `outputs/`: splits, metrics, plots, and timestamped run directories (created at runtime)

## Requirements

- Python **3.10+** recommended
- PyTorch / TorchVision (see pinned versions in `requirements.txt`)

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download CIFAR-10:

```bash
python download_data.py
```

## Quickstart

### Baseline (independent local training)

Runs a simple baseline where each client trains from the same initialized weights, without server aggregation.

```bash
python run_baseline.py
```

Outputs (created if missing):
- `outputs/splits/`: cached Dirichlet splits (e.g., `cifar10_dirichlet_a0.1_s42_c10.npy`)
- `outputs/metrics/baseline_results.csv`: per-client evaluation metrics

### FL (Flower simulation)

```bash
python run_fl.py
```

This creates a timestamped run directory such as `outputs/RUN_YYYYMMDD_HHMMSS/` containing metrics and plots.

### PFL (prototype exchange)

```bash
python run_pfl.py
```

## Configuration (environment variables)

The main scripts are configured through environment variables (defaults shown):

- **Experiment size**
  - `NUM_CLIENTS` (default: `10`)
  - `NUM_ROUNDS` (default: `2`)
  - `EPOCHS` (default: `5`)
- **Non-IID split**
  - `ALPHA` (default: `0.1`) — Dirichlet concentration parameter
  - `SEED` (default: `42`)
- **Device selection**
  - `DEVICE` (optional): force a device string (e.g. `cpu`, `cuda`, `mps`)
  - If not set, the code auto-selects `cuda`/`mps` when available, else `cpu`
- **Training options**
  - `AUGMENTATION` (default: `off`) — set to `on` to enable RandAugment path where supported
  - `MIXUP_ALPHA` (default depends on script) — mixup strength when augmentation is enabled
  - `LAMBDA_P` / `LD` (default: `0.1`) — prototype loss weight
  - `USE_CLASS_WEIGHTS` (default: `1`) — enable/disable class-weighted loss for non-IID robustness (used in PFL/FL clients)
  - `MAX_SAMPLES_PER_CLIENT` (default: `0`) — if >0, truncates per-client training set for quick runs
  - `PROGRESS` (default: `0`) — set to `1` for verbose per-client progress
- **Security logging**
  - `SECURITY_LOGGING` (default: `1`) — enables run-level security logging under the timestamped `outputs/RUN_.../`

Example run:

```bash
NUM_CLIENTS=20 NUM_ROUNDS=5 EPOCHS=2 ALPHA=0.3 SEED=123 DEVICE=mps python run_pfl.py
```

## Reproducibility notes

- Splits are cached under `outputs/splits/` and re-used when the exact `(alpha, seed, num_clients)` match.
- The Flower simulations create **timestamped** output folders to avoid accidental overwrites.

## Colab notebooks

The `colab/` folder includes notebooks for common experiment workflows (baseline runs, DP variants, and security evaluations). If you prefer notebook execution, start there.

## Project status

This is an experimental research codebase. Interfaces and default hyperparameters may change as experiments evolve.

## License

No `LICENSE` file is included in this repository. If you plan to open-source or distribute this project, add an explicit license.
