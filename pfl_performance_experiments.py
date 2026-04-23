"""
pfl_performance_experiments.py
-------------------------------
Three-phase hyperparameter sweep for Prototype FL.

Phase 1 — Lambda × Class-Weight grid (2 rounds, epochs=5, no augment)
  Runs 1–8: all combinations of lambda_p in {0.01, 0.05, 0.10, 0.30}
            × class_weights in {off, on}

Phase 2 — Epoch sweep (best lambda + class_weights from Phase 1, no augment)
  Runs 9–11: epochs in {3, 5, 10}

Phase 3 — Augmentation (best lambda + class_weights + best epochs from Phase 2)
  Run 12: RandAugment + Mixup  on
  Run 13: no augmentation      (may already exist from Phase 2)

After each phase the best configuration is printed and carried forward automatically.

Usage (from project root):
    python pfl_performance_experiments.py --phase 1
    python pfl_performance_experiments.py --phase 2
    python pfl_performance_experiments.py --phase 3
    python pfl_performance_experiments.py --phase all   # run all three sequentially

All outputs → outputs/metrics/hparam_sweep/phase{1|2|3}/

Environment variables (all optional \u2014 no .env file required):
    NUM_CLIENTS   number of FL clients            (default 10)
    NUM_ROUNDS    rounds per run, all phases       (default 2)
    ALPHA         Dirichlet alpha for data split   (default 0.1)
    SEED          global random seed               (default 42)
    DEVICE        cuda / mps / cpu (auto-detected if unset)
    RAY_NUM_GPUS  GPU fraction per Ray actor       (default 0.2, CUDA only)
"""

from __future__ import annotations



import argparse
import csv
import os
import shutil
from typing import Dict, List, Optional, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import numpy as np
import torch

from src.data_utils import (
    Cifar10Config,
    DirichletSplitConfig,
    dirichlet_split_indices,
    load_cifar10,
    load_split,
)
from src.train_utils import set_seed

try:
    import flwr as fl  # type: ignore
except Exception:
    fl = None  # type: ignore


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def _select_device() -> str:
    requested = os.environ.get("DEVICE", "")
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Ray / checkpoint helpers
# ---------------------------------------------------------------------------

def _reset_ray_if_needed() -> None:
    try:
        import ray  # type: ignore
        if ray.is_initialized():
            ray.shutdown()
    except Exception:
        pass


def _clear_client_model_checkpoints() -> None:
    d = os.path.join("outputs", "client_models")
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------------------
# Single-run delegate
# ---------------------------------------------------------------------------

def _run(
    *,
    label: str,
    epochs: int,
    lambda_p: float,
    use_class_weights: bool,
    train_transform: str,
    mixup_alpha: float,
    num_clients: int,
    num_rounds: int,
    alpha: float,
    seed: int,
    device: str,
    train_ds,
    test_ds,
    split,
    out_dir: str,
) -> Dict[str, float]:
    """
    Run one FL simulation and return the final-round avg accuracies.
    Returns dict with keys: avg_global, avg_local_proportional, avg_local.
    """
    from run_pfl import run_flower_experiment

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "results.csv")
    plot_path = os.path.join(out_dir, "accuracy_curve.png")

    _reset_ray_if_needed()
    _clear_client_model_checkpoints()

    # Communicate lambda_p and class_weight flag through environment so that
    # FlowerPrototypeClient and train_local_proto pick them up unchanged.
    os.environ["LAMBDA_P"] = str(lambda_p)
    os.environ["LD"] = str(lambda_p)                            # alias used in FlowerPrototypeClient
    os.environ["USE_CLASS_WEIGHTS"] = "1" if use_class_weights else "0"

    print(
        f"\n{'='*70}\n"
        f"  Run: {label}\n"
        f"  lambda_p={lambda_p}  class_weights={'on' if use_class_weights else 'off'}\n"
        f"  epochs={epochs}  augment={train_transform}  mixup={mixup_alpha}\n"
        f"  rounds={num_rounds}  clients={num_clients}\n"
        f"  out_dir={out_dir}\n"
        f"{'='*70}\n",
        flush=True,
    )

    run_flower_experiment(
        exp_mode=label,
        epochs=epochs,
        train_transform=train_transform,
        mixup_alpha=mixup_alpha,
        num_clients=num_clients,
        num_rounds=num_rounds,
        alpha=alpha,
        seed=seed,
        device=device,
        train_ds=train_ds,
        test_ds=test_ds,
        split=split,
        metrics_csv_path=csv_path,
        plot_path=plot_path,
        use_class_weights=use_class_weights,
        lambda_p=lambda_p,
    )

    # Read the last row of the CSV for final-round averages
    return _read_last_row(csv_path)


def _read_last_row(csv_path: str) -> Dict[str, float]:
    """Return the last data row of the results CSV as a float dict."""
    if not os.path.isfile(csv_path):
        return {"avg_global": 0.0, "avg_local_proportional": 0.0, "avg_local": 0.0}
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {"avg_global": 0.0, "avg_local_proportional": 0.0, "avg_local": 0.0}
    last = rows[-1]
    return {
        "avg_global": float(last.get("avg_global", 0)),
        "avg_local_proportional": float(last.get("avg_local_proportional", 0)),
        "avg_local": float(last.get("avg_local", 0)),
    }


# ---------------------------------------------------------------------------
# Summary CSV writer
# ---------------------------------------------------------------------------

def _write_summary(path: str, rows: List[dict]) -> None:
    """Append experiment rows to a phase-level summary CSV."""
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = list(rows[0].keys())
    file_exists = os.path.isfile(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            w.writeheader()
        w.writerows(rows)


def _print_summary_table(rows: List[dict], title: str) -> None:
    print(f"\n{'='*90}")
    print(f"  {title}")
    print(f"{'='*90}")
    if not rows:
        print("  (no results)")
        return
    headers = list(rows[0].keys())
    widths = [max(len(str(r.get(h, ""))) for r in rows + [{"dummy": h}]) for h in headers]
    widths = [max(w, len(h)) for w, h in zip(widths, headers)]
    fmt = "  " + "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print("  " + "  ".join("-" * w for w in widths))
    for r in rows:
        print(fmt.format(*[str(r.get(h, "")) for h in headers]))
    print(f"{'='*90}\n")


def _best_row(rows: List[dict], key: str = "avg_local") -> dict:
    """Return the row with the highest value for `key`."""
    return max(rows, key=lambda r: float(r.get(key, 0)))


# ---------------------------------------------------------------------------
# Phase 1 — Lambda × Class-Weight grid
# ---------------------------------------------------------------------------

PHASE1_LAMBDAS = [0.01, 0.05, 0.10, 0.30]
PHASE1_CLASS_WEIGHTS = [False, True]


def run_phase1(
    *,
    num_clients: int,
    num_rounds: int,
    alpha: float,
    seed: int,
    device: str,
    train_ds,
    test_ds,
    split,
    base_dir: str,
) -> dict:
    """
    Phase 1: lambda × class_weight grid.
    Runs 8 configurations (fixed epochs=5, no augment, 1 round).
    Returns the best config dict.
    """
    print("\n" + "#" * 80)
    print("# PHASE 1 — Lambda × Class-Weight Sweep")
    print("# Fixed: epochs=5, augment=off, 1 round")
    print("#" * 80)

    summary_rows: List[dict] = []
    run_id = 1

    for lp in PHASE1_LAMBDAS:
        for cw in PHASE1_CLASS_WEIGHTS:
            label = f"p1_run{run_id}_lp{lp}_cw{'on' if cw else 'off'}"
            out_dir = os.path.join(base_dir, "phase1", label)
            accs = _run(
                label=label,
                epochs=5,
                lambda_p=lp,
                use_class_weights=cw,
                train_transform="default",
                mixup_alpha=0.0,
                num_clients=num_clients,
                num_rounds=num_rounds,
                alpha=alpha,
                seed=seed,
                device=device,
                train_ds=train_ds,
                test_ds=test_ds,
                split=split,
                out_dir=out_dir,
            )
            row = {
                "run_id": run_id,
                "lambda_p": lp,
                "class_weights": "on" if cw else "off",
                "epochs": 5,
                "avg_global": f"{accs['avg_global']:.4f}",
                "avg_local_prop": f"{accs['avg_local_proportional']:.4f}",
                "avg_local": f"{accs['avg_local']:.4f}",
            }
            summary_rows.append(row)
            run_id += 1

    summary_path = os.path.join(base_dir, "phase1_summary.csv")
    _write_summary(summary_path, summary_rows)
    _print_summary_table(summary_rows, "Phase 1 Results — Lambda × Class-Weight")

    best = _best_row(summary_rows, "avg_local")
    print(
        f"  ✓ Best Phase 1 config → run {best['run_id']} | "
        f"lambda_p={best['lambda_p']}  class_weights={best['class_weights']}  "
        f"avg_local={best['avg_local']}\n"
    )
    return {
        "lambda_p": float(best["lambda_p"]),
        "use_class_weights": best["class_weights"] == "on",
    }


# ---------------------------------------------------------------------------
# Phase 2 — Epoch sweep
# ---------------------------------------------------------------------------

PHASE2_EPOCHS = [3, 5, 10]


def run_phase2(
    *,
    best_lambda_p: float,
    best_use_class_weights: bool,
    num_clients: int,
    num_rounds: int,
    alpha: float,
    seed: int,
    device: str,
    train_ds,
    test_ds,
    split,
    base_dir: str,
) -> dict:
    """
    Phase 2: epoch sweep with the best lambda + class_weights from Phase 1.
    Runs 3 configurations (epochs in {3, 5, 10}, no augment).
    Returns the best config dict.
    """
    print("\n" + "#" * 80)
    print("# PHASE 2 — Epoch Sweep")
    print(f"# Fixed: lambda_p={best_lambda_p}  class_weights={'on' if best_use_class_weights else 'off'}  augment=off")
    print("#" * 80)

    summary_rows: List[dict] = []
    run_id = 9

    for ep in PHASE2_EPOCHS:
        label = f"p2_run{run_id}_ep{ep}"
        out_dir = os.path.join(base_dir, "phase2", label)
        accs = _run(
            label=label,
            epochs=ep,
            lambda_p=best_lambda_p,
            use_class_weights=best_use_class_weights,
            train_transform="default",
            mixup_alpha=0.0,
            num_clients=num_clients,
            num_rounds=num_rounds,
            alpha=alpha,
            seed=seed,
            device=device,
            train_ds=train_ds,
            test_ds=test_ds,
            split=split,
            out_dir=out_dir,
        )
        row = {
            "run_id": run_id,
            "lambda_p": best_lambda_p,
            "class_weights": "on" if best_use_class_weights else "off",
            "epochs": ep,
            "avg_global": f"{accs['avg_global']:.4f}",
            "avg_local_prop": f"{accs['avg_local_proportional']:.4f}",
            "avg_local": f"{accs['avg_local']:.4f}",
        }
        summary_rows.append(row)
        run_id += 1

    summary_path = os.path.join(base_dir, "phase2_summary.csv")
    _write_summary(summary_path, summary_rows)
    _print_summary_table(summary_rows, "Phase 2 Results — Epoch Sweep")

    best = _best_row(summary_rows, "avg_local")
    print(
        f"  ✓ Best Phase 2 config → run {best['run_id']} | "
        f"epochs={best['epochs']}  avg_local={best['avg_local']}\n"
    )
    return {"epochs": int(best["epochs"])}


# ---------------------------------------------------------------------------
# Phase 3 — Augmentation comparison
# ---------------------------------------------------------------------------

def run_phase3(
    *,
    best_lambda_p: float,
    best_use_class_weights: bool,
    best_epochs: int,
    num_clients: int,
    num_rounds: int,
    alpha: float,
    seed: int,
    device: str,
    train_ds_default,     # loaded with train_transform="default"
    train_ds_augmented,   # loaded with train_transform="randaugment"
    test_ds,
    split,
    base_dir: str,
) -> None:
    """
    Phase 3: augmentation comparison.
    Run 12: RandAugment + Mixup
    Run 13: no augmentation
    """
    print("\n" + "#" * 80)
    print("# PHASE 3 — Augmentation Comparison")
    print(f"# Fixed: lambda_p={best_lambda_p}  class_weights={'on' if best_use_class_weights else 'off'}  epochs={best_epochs}")
    print("#" * 80)

    summary_rows: List[dict] = []

    configs = [
        # (run_id, label, train_ds, train_transform, mixup_alpha)
        (12, "p3_run12_augment_on",  train_ds_augmented, "randaugment", 0.2),
        (13, "p3_run13_augment_off", train_ds_default,   "default",     0.0),
    ]

    for run_id, label, ds, tf, mixup in configs:
        out_dir = os.path.join(base_dir, "phase3", label)
        accs = _run(
            label=label,
            epochs=best_epochs,
            lambda_p=best_lambda_p,
            use_class_weights=best_use_class_weights,
            train_transform=tf,
            mixup_alpha=mixup,
            num_clients=num_clients,
            num_rounds=num_rounds,
            alpha=alpha,
            seed=seed,
            device=device,
            train_ds=ds,
            test_ds=test_ds,
            split=split,
            out_dir=out_dir,
        )
        row = {
            "run_id": run_id,
            "lambda_p": best_lambda_p,
            "class_weights": "on" if best_use_class_weights else "off",
            "epochs": best_epochs,
            "augmentation": "on" if tf == "randaugment" else "off",
            "avg_global": f"{accs['avg_global']:.4f}",
            "avg_local_prop": f"{accs['avg_local_proportional']:.4f}",
            "avg_local": f"{accs['avg_local']:.4f}",
        }
        summary_rows.append(row)

    summary_path = os.path.join(base_dir, "phase3_summary.csv")
    _write_summary(summary_path, summary_rows)
    _print_summary_table(summary_rows, "Phase 3 Results — Augmentation Comparison")

    best = _best_row(summary_rows, "avg_local")
    print(
        f"  ✓ Best Phase 3 config → run {best['run_id']} | "
        f"augmentation={best['augmentation']}  avg_local={best['avg_local']}\n"
    )


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="3-phase PFL hyperparameter sweep.")
    p.add_argument(
        "--phase",
        required=True,
        choices=("1", "2", "3", "all"),
        help=(
            "1: lambda×class-weight grid | "
            "2: epoch sweep | "
            "3: augmentation comparison | "
            "all: run all three sequentially"
        ),
    )
    # Allow manually overriding the 'best' config when running phases individually
    p.add_argument("--lambda-p", type=float, default=None, help="Override best lambda_p (for phase 2/3).")
    p.add_argument("--class-weights", choices=("on", "off"), default=None, help="Override class-weights (for phase 2/3).")
    p.add_argument("--epochs", type=int, default=None, help="Override best epochs (for phase 3).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    num_clients = int(os.environ.get("NUM_CLIENTS", "10"))
    alpha = float(os.environ.get("ALPHA", "0.1"))
    seed = int(os.environ.get("SEED", "42"))
    device = _select_device()

    # All phases use the same number of rounds (default 2 to properly simulate FL)
    num_rounds = int(os.environ.get("NUM_ROUNDS", "2"))


    set_seed(seed)

    # Load dataset variants
    train_ds_default, test_ds = load_cifar10(Cifar10Config(root="data", train_transform="default"))
    train_ds_augmented, _ = load_cifar10(Cifar10Config(root="data", train_transform="randaugment"))

    split_path = f"outputs/splits/cifar10_dirichlet_a{alpha}_s{seed}_c{num_clients}.npy"
    if not os.path.exists(split_path):
        cfg_s = DirichletSplitConfig(num_clients=num_clients, alpha=alpha, seed=seed)
        split = dirichlet_split_indices(train_ds_default, cfg_s)
    else:
        split = load_split(split_path)

    base_dir = os.path.join("outputs", "metrics", "hparam_sweep")
    os.makedirs(base_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Determine which phases to run and carry best config forward
    # -----------------------------------------------------------------------
    phases = ["1", "2", "3"] if args.phase == "all" else [args.phase]

    # Defaults / CLI overrides for carrying best config between phases
    best_lambda_p: float = args.lambda_p if args.lambda_p is not None else 0.10
    best_use_cw: bool = (args.class_weights == "on") if args.class_weights is not None else True
    best_epochs: int = args.epochs if args.epochs is not None else 5

    if "1" in phases:
        p1_best = run_phase1(
            num_clients=num_clients,
            num_rounds=num_rounds,
            alpha=alpha,
            seed=seed,
            device=device,
            train_ds=train_ds_default,
            test_ds=test_ds,
            split=split,
            base_dir=base_dir,
        )
        # Only override if not user-specified
        if args.lambda_p is None:
            best_lambda_p = p1_best["lambda_p"]
        if args.class_weights is None:
            best_use_cw = p1_best["use_class_weights"]

    if "2" in phases:
        p2_best = run_phase2(
            best_lambda_p=best_lambda_p,
            best_use_class_weights=best_use_cw,
            num_clients=num_clients,
            num_rounds=num_rounds,
            alpha=alpha,
            seed=seed,
            device=device,
            train_ds=train_ds_default,
            test_ds=test_ds,
            split=split,
            base_dir=base_dir,
        )
        if args.epochs is None:
            best_epochs = p2_best["epochs"]

    if "3" in phases:
        run_phase3(
            best_lambda_p=best_lambda_p,
            best_use_class_weights=best_use_cw,
            best_epochs=best_epochs,
            num_clients=num_clients,
            num_rounds=num_rounds,
            alpha=alpha,
            seed=seed,
            device=device,
            train_ds_default=train_ds_default,
            train_ds_augmented=train_ds_augmented,
            test_ds=test_ds,
            split=split,
            base_dir=base_dir,
        )


if __name__ == "__main__":
    main()
