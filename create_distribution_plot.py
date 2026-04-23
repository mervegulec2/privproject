import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_utils import load_cifar10, Cifar10Config, DirichletSplitConfig, dirichlet_split_indices

def create_plot():
    # 1. Configuration
    cifar_cfg = Cifar10Config(root="data")
    split_cfg = DirichletSplitConfig(num_clients=10, alpha=0.1, seed=42)
    
    # 2. Load Data
    print("Loading CIFAR-10...")
    train_ds, _ = load_cifar10(cifar_cfg)
    targets = np.array(train_ds.targets)
    
    # 3. Partition Data
    print(f"Partitioning data (alpha={split_cfg.alpha}, seed={split_cfg.seed})...")
    client_map = dirichlet_split_indices(train_ds, split_cfg)
    
    # 4. Calculate Distributions
    # Matrix of size (num_clients, num_classes)
    dist = np.zeros((split_cfg.num_clients, split_cfg.num_classes))
    for cid, indices in client_map.items():
        client_targets = targets[indices]
        unique, counts = np.unique(client_targets, return_counts=True)
        for u, c in zip(unique, counts):
            dist[cid, u] = c
            
    # Normalize by client total size to get proportions (as in the reference image)
    dist_norm = dist / dist.sum(axis=1, keepdims=True)
    
    # 5. Plotting
    print("Generating plot...")
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="white")
    
    # Colors for 10 classes
    colors = sns.color_palette("tab10", n_colors=10)
    
    clients = [f"Client {i+1}" for i in range(split_cfg.num_clients)]
    left = np.zeros(split_cfg.num_clients)
    
    for class_idx in range(split_cfg.num_classes):
        plt.barh(clients, dist_norm[:, class_idx], left=left, color=colors[class_idx], label=f"Class {class_idx}")
        left += dist_norm[:, class_idx]
        
    plt.title(f"Class Distribution per Client (Dirichlet, α={split_cfg.alpha})", fontsize=15)
    plt.xlabel("Class Distribution", fontsize=12)
    plt.ylabel("Client", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Classes")
    plt.tight_layout()
    
    # 6. Save
    os.makedirs("outputs", exist_ok=True)
    output_path = "outputs/class_distribution.png"
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    create_plot()
