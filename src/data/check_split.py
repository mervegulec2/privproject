"""Generate and inspect Dirichlet non-IID splits for CIFAR-10. Default alpha=0.1 for stronger heterogeneity."""
from __future__ import annotations

import argparse
from src.data.cifar import Cifar10Config, load_cifar10
from src.data.split import DirichletSplitConfig, dirichlet_split_indices, class_hist, save_split

LABELS = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def main():
    ap = argparse.ArgumentParser(description="Generate Dirichlet splits for CIFAR-10 (non-IID).")
    ap.add_argument("--alpha", type=float, default=0.1,
                    help="Dirichlet concentration (default 0.1 = stronger non-IID, fewer classes per client).")
    ap.add_argument("--seeds", type=str, default="42,123,999", help="Comma-separated seeds, e.g. 42,123,999")
    ap.add_argument("--min_size", type=int, default=500, help="Min samples per client")
    args = ap.parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    train_ds, _ = load_cifar10(Cifar10Config(root="data"))

    for seed in seeds:
        cfg = DirichletSplitConfig(
            num_clients=15,
            alpha=args.alpha,
            seed=seed,
            min_size_per_client=args.min_size,
        )
        client_map = dirichlet_split_indices(train_ds, cfg)

        print(f"\n=== Split seed={seed}, alpha={cfg.alpha}, clients={cfg.num_clients} ===")
        total = 0
        for cid in range(cfg.num_clients):
            idx = client_map[cid]
            total += len(idx)
            hist = class_hist(train_ds, idx, num_classes=10)
            present = int((hist > 0).sum())
            top3 = sorted([(LABELS[i], int(hist[i])) for i in range(10)], key=lambda x: -x[1])[:3]
            print(f"Client {cid:02d}: n={len(idx):4d}, classes_present={present:2d}, top3={top3}")

        print("Total assigned:", total)

        out_path = f"outputs/splits/cifar10_dirichlet_a{args.alpha}_seed{seed}.npy"
        save_split(client_map, out_path)
        print("Saved split:", out_path)


if __name__ == "__main__":
    main()