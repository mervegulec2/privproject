import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Ensure current directory is in path for imports
sys.path.append(os.getcwd())

from src.data_utils import load_cifar10, Cifar10Config, load_split

def plot_client_distributions(split_path, save_path="outputs/metrics/class_distribution.png"):
    """
    Visualizes the client data distribution (Non-IID) as a stacked bar chart.
    """
    print(f"Loading data split from: {split_path}")
    
    # 1. Load Data
    train_ds, _ = load_cifar10(Cifar10Config(root="data"))
    targets = np.array(train_ds.targets)
    split = load_split(split_path)
    
    num_clients = len(split)
    num_classes = 10
    
    # 2. Calculate distribution
    distribution = np.zeros((num_clients, num_classes))
    for cid, indices in split.items():
        labels = targets[indices]
        unique, counts = np.unique(labels, return_counts=True)
        for u, c in zip(unique, counts):
            distribution[cid, u] = c
            
    # Normalize
    distribution_norm = distribution / (distribution.sum(axis=1, keepdims=True) + 1e-8)
    
    # 3. Plotting
    plt.figure(figsize=(12, 7))
    clients = [f"Client {i+1}" for i in range(num_clients)]
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
    
    left = np.zeros(num_clients)
    for c in range(num_classes):
        plt.barh(clients, distribution_norm[:, c], left=left, label=f"Class {c}", color=colors[c])
        left += distribution_norm[:, c]
        
    plt.xlabel("Class Distribution")
    plt.ylabel("Client")
    plt.title("Class Distribution per Client (Dirichlet, α=0.1)")
    plt.legend(title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # 4. Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"Success! Figure saved to: {save_path}")

if __name__ == "__main__":
    # Actual split file from your PFL runs
    split_file = "outputs/splits/cifar10_dirichlet_a0.1_s42_c10.npy"
    
    if os.path.exists(split_file):
        plot_client_distributions(split_file)
    else:
        print(f"Error: {split_file} not found. Please run PFL training first to generate the split.")
