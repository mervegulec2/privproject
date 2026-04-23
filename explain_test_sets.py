import os
import numpy as np
from src.data_utils import load_cifar10, Cifar10Config, load_split
from src.eval.test_sets import (
    create_local_aware_indices,
    create_local_proportional_indices
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

    # 1. Global Original CIFAR10 Test Set
    print(f"\n[Set 1] Global Original: Total {len(test_ds)} samples.")
    classes, counts = np.unique(test_targets, return_counts=True)
    print(f"        Class distribution: {dict(zip(classes, counts))}")

    # print 3 clients at most
    for cid in range(min(num_clients, 3)):
        print(f"\\n" + "-"*50)
        print(f"Client {cid}")
        print("-" * 50)
        
        # Client train data 
        train_targets = np.array(train_ds.targets)
        client_train_labels = train_targets[split[cid]]
        t_classes, t_counts = np.unique(client_train_labels, return_counts=True)
        t_props = {c: f"{(count/len(client_train_labels))*100:.1f}%" for c, count in zip(t_classes, t_counts)}
        print(f"[A] Train data: Total {len(client_train_labels)} samples")
        for cls, count in zip(t_classes, t_counts):
            print(f"    Class {cls}: {count} sample ({t_props[cls]})")
            
        seen_classes = set(t_classes)

        # 2. Train proportional test set
        lp_indices = create_local_proportional_indices(test_ds, train_ds, split[cid], total_test_samples=1000, seed=seed)
        lp_targets = test_targets[lp_indices]
        classes_p, counts_p = np.unique(lp_targets, return_counts=True)
        p_props = {c: f"{(count/len(lp_targets))*100:.1f}%" for c, count in zip(classes_p, counts_p)}
        print(f"\\n[Case 2] Train proportional test set: Total {len(lp_indices)} samples")
        for cls, count in zip(classes_p, counts_p):
            print(f"    Class {cls}: {count} test samples ({p_props.get(cls, '0%')})")

        # 3. Complete data from CIFAR10 test set that client has seen during training
        la_indices = create_local_aware_indices(test_ds, seen_classes)
        la_targets = test_targets[la_indices]
        classes, counts = np.unique(la_targets, return_counts=True)
        print(f"[Set 3] Local-Aware: Total {len(la_indices)} samples.")
        for cls, count in zip(classes_a, counts_a):
            print(f"    Class {cls}: {count} test samples")

if __name__ == "__main__":
    explain()
