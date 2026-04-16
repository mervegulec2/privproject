"""
inspect_splits.py
-----------------
Prints the class distribution for each client under a given
(seed, alpha, num_clients) configuration.

Usage:
    python inspect_splits.py                  # uses defaults below
    python inspect_splits.py --seed 123       # override seed
    python inspect_splits.py --alpha 0.5      # override alpha
"""

import argparse
import os

from src.data_utils import (
    Cifar10Config,
    DirichletSplitConfig,
    load_cifar10,
    dirichlet_split_indices,
    save_split,
    load_split,
    print_client_distributions,
)


def main():
    parser = argparse.ArgumentParser(description="Inspect client data distributions.")
    parser.add_argument("--seed",        type=int,   default=42,   help="Random seed (default: 42)")
    parser.add_argument("--alpha",       type=float, default=0.1,  help="Dirichlet alpha (default: 0.1)")
    parser.add_argument("--num_clients", type=int,   default=10,   help="Number of clients (default: 10)")
    args = parser.parse_args()

    # Load dataset
    train_ds, _ = load_cifar10(Cifar10Config(root="data"))

    # Load or generate split
    split_dir = "outputs/splits"
    os.makedirs(split_dir, exist_ok=True)
    split_path = os.path.join(
        split_dir,
        f"cifar10_dirichlet_a{args.alpha}_s{args.seed}_c{args.num_clients}.npy"
    )

    if os.path.exists(split_path):
        print(f"Loading existing split from: {split_path}")
        split = load_split(split_path)
    else:
        print(f"Generating new split (seed={args.seed}, alpha={args.alpha}, clients={args.num_clients})...")
        cfg = DirichletSplitConfig(
            num_clients=args.num_clients,
            alpha=args.alpha,
            seed=args.seed,
        )
        split = dirichlet_split_indices(train_ds, cfg)
        save_split(split, split_path)
        print(f"Split saved to: {split_path}")

    # Print distributions
    print(f"\nConfiguration — seed={args.seed}, alpha={args.alpha}, num_clients={args.num_clients}")
    print_client_distributions(split, train_ds)


if __name__ == "__main__":
    main()
