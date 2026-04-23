import os
import numpy as np
from src.data_utils import load_cifar10, Cifar10Config, load_split

def analyze_client_classes():
    num_clients = 10
    alpha = 0.1
    seed = 42
    split_path = f"outputs/splits/cifar10_dirichlet_a{alpha}_s{seed}_c{num_clients}.npy"
    
    if not os.path.exists(split_path):
        print(f"Split path {split_path} not found.")
        return

    train_ds, _ = load_cifar10(Cifar10Config())
    targets = np.array(train_ds.targets)
    split = load_split(split_path)

    for cid, indices in split.items():
        client_targets = targets[indices]
        classes, counts = np.unique(client_targets, return_counts=True)
        print(f"Client {cid}: Classes: {classes}, Counts: {counts}")

if __name__ == "__main__":
    analyze_client_classes()
