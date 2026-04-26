import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse

# Ensure src is in path
sys.path.append(os.getcwd())

from src.data_utils import load_cifar10, Cifar10Config, load_split

def plot_distribution_pro(seed=42, alpha=0.1, num_clients=10):
    # 1. Load Data and Split
    train_ds, _ = load_cifar10(Cifar10Config(root="data"))
    targets = np.array(train_ds.targets)
    
    split_file = f"outputs/splits/cifar10_dirichlet_a{alpha}_s{seed}_c{num_clients}.npy"
    if not os.path.exists(split_file):
        print(f"Error: Split file {split_file} not found. Please run inspect_splits.py first to generate it.")
        return

    split = load_split(split_file)
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    # 2. Calculate Distribution Matrix
    distribution = np.zeros((num_clients, 10))
    for cid, indices in split.items():
        labels = targets[indices]
        unique, counts = np.unique(labels, return_counts=True)
        for u, c in zip(unique, counts):
            distribution[cid, u] = c
            
    # Normalize
    distribution_norm = distribution / (distribution.sum(axis=1, keepdims=True) + 1e-8)
    
    # 3. Plotting
    plt.figure(figsize=(14, 8))
    # We use Client 0-9 to match your terminal table exactly
    clients = [f"Client {i}" for i in range(num_clients)]
    
    # Using a professional color palette
    colors = plt.cm.get_cmap('tab10', 10)(np.linspace(0, 1, 10))
    
    left = np.zeros(num_clients)
    for c in range(10):
        plt.barh(clients, distribution_norm[:, c], left=left, label=class_names[c], color=colors[c], edgecolor='white', linewidth=0.5)
        left += distribution_norm[:, c]
        
    plt.xlabel("Proportion of Samples", fontsize=12)
    plt.ylabel("Client ID", fontsize=12)
    plt.title(f"Class Distribution per Client (Dirichlet alpha={alpha}, seed={seed})", fontsize=14, pad=20)
    
    # Legend outside to the right
    plt.legend(title="Classes", bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
    
    # Reverse y-axis to show Client 0 at the top (to match table)
    plt.gca().invert_yaxis()
    
    plt.tight_layout()
    
    save_path = f"outputs/metrics/class_dist_a{alpha}_s{seed}.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Professional distribution plot saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--num_clients", type=int, default=10)
    args = parser.parse_args()
    
    plot_distribution_pro(args.seed, args.alpha, args.num_clients)
