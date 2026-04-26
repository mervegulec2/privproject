import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Ensure src is in path
sys.path.append(os.getcwd())
from src.data_utils import load_cifar10, Cifar10Config, load_split

def plot_evaluation_protocol_final(split_path, client_id=1, save_path="outputs/metrics/eval_protocol_final.png"):
    # 1. Load Data
    train_ds, test_ds = load_cifar10(Cifar10Config(root="data"))
    train_targets = np.array(train_ds.targets)
    split = load_split(split_path)
    
    # 2. Get distributions
    client_indices = split[client_id]
    client_train_labels = train_targets[client_indices]
    present_classes = np.unique(client_train_labels)
    
    # - Global Test Dist (Uniform 10%)
    global_dist = np.ones(10) / 10.0
    
    # - Local-Aware Dist (Uniform among present classes)
    aware_dist = np.zeros(10)
    for c in present_classes:
        aware_dist[c] = 1.0 / len(present_classes)
        
    # - Local-Proportional Dist (Matches training proportions)
    prop_dist = np.zeros(10)
    unique, counts = np.unique(client_train_labels, return_counts=True)
    for u, c in zip(unique, counts):
        prop_dist[u] = c / len(client_train_labels)

    # 3. Plotting
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    # Using 'Class 0' naming to match your request
    classes = [f"Class {i}" for i in range(10)]
    
    titles = ["1. Global Test Set", "2. Local-Aware Test", "3. Local-Proportional Test"]
    data = [global_dist, aware_dist, prop_dist]
    
    for ax, dist, title in zip(axes, data, titles):
        ax.bar(classes, dist, color=colors, edgecolor='black', alpha=0.8)
        ax.set_title(title, fontweight='bold')
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("Classes")
        # Rotate x-labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45)
        if ax == axes[0]: ax.set_ylabel("Probability Density")
    
    # Match client naming to terminal table (Client 1 stays Client 1)
    plt.suptitle(f"Evaluation Protocol Visualization for Client {client_id}", fontsize=14, y=1.05)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Final evaluation figure saved to: {save_path}")

if __name__ == "__main__":
    split_file = "outputs/splits/cifar10_dirichlet_a0.1_s42_c10.npy"
    if os.path.exists(split_file):
        # Client 1 as requested (matches terminal table)
        plot_evaluation_protocol_final(split_file, client_id=1) 
    else:
        print("Error: Split file not found.")
