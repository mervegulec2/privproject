import os
import numpy as np
from torch.utils.data import Subset
from src.data_utils import load_cifar10, Cifar10Config, load_split, get_seen_classes
from src.eval.test_sets import (
    create_balanced_test_indices, 
    create_local_aware_indices, 
    create_local_aware_balanced_indices
)

def explain():
    num_clients = 10
    alpha = 0.1
    seed = 42
    split_path = f"outputs/splits/cifar10_dirichlet_a{alpha}_s{seed}_c{num_clients}.npy"
    
    if not os.path.exists(split_path):
        print("Split file not found. Please run the simulation once or generate the split.")
        return

    train_ds, test_ds = load_cifar10(Cifar10Config())
    test_targets = np.array(test_ds.targets)
    split = load_split(split_path)

    print("="*60)
    print("TEST SET CONFIGURATIONS AND CLASS DISTRIBUTIONS")
    print("="*60)

    # 1. Global Original
    print(f"\n[Set 1] Global Original: Total {len(test_ds)} samples.")

    # 2. Global Balanced
    gb_indices = create_balanced_test_indices(test_ds, samples_per_class=100, seed=seed)
    print(f"[Set 2] Global Balanced: Total {len(gb_indices)} samples (approx 100 per class).")

    for cid in range(num_clients):
        print(f"\n--- Client {cid} ---")
        seen_classes = get_seen_classes(train_ds, split[cid])
        print(f"Classes seen in training: {sorted(list(seen_classes))}")

        # 3. Local Aware
        la_indices = create_local_aware_indices(test_ds, seen_classes)
        la_targets = test_targets[la_indices]
        classes, counts = np.unique(la_targets, return_counts=True)
        print(f"[Set 3] Local-Aware: Total {len(la_indices)} samples.")
        print(f"        Class distribution: {dict(zip(classes, counts))}")

        # 4. Local Balanced
        lab_indices = create_local_aware_balanced_indices(test_ds, seen_classes, samples_per_class=100, seed=seed)
        lab_targets = test_targets[lab_indices]
        classes_b, counts_b = np.unique(lab_targets, return_counts=True)
        print(f"[Set 4] Local-Balanced: Total {len(lab_indices)} samples.")
        print(f"        Class distribution: {dict(zip(classes_b, counts_b))}")

if __name__ == "__main__":
    explain()
