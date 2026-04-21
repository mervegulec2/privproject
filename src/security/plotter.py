import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List

def plot_privacy_utility_tradeoff(
    noise_levels: List[float], 
    accuracies: List[float], 
    psnr_scores: List[float], 
    save_path: str = "outputs/security/tradeoff_plot.png"
):
    """
    Creates an academic-style plot showing the trade-off between Utility (Accuracy)
    and Privacy (Reconstruction Quality / PSNR).
    X-axis: Defense Strength (e.g., DP Noise Multiplier)
    Y1 (Left): Model Accuracy (Higher is better for utility)
    Y2 (Right): PSNR (Lower is better for privacy)
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    sns.set_theme(style="whitegrid", rc={"axes.edgecolor": "black", "grid.color": "lightgrey"})

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # X-axis setup
    x = np.arange(len(noise_levels))
    x_labels = [str(nl) if nl > 0 else "Baseline" for nl in noise_levels]

    # --- Utility Plot (Left Axis) ---
    color = 'tab:blue'
    ax1.set_xlabel('Defense Strength (Noise Scale $\sigma$)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Utility: Accuracy (%)', color=color, fontsize=12, fontweight='bold')
    ax1.plot(x, accuracies, marker='o', color=color, linewidth=2, markersize=8, label='Accuracy')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels)
    ax1.set_ylim([max(0, min(accuracies) - 10), min(100, max(accuracies) + 5)])

    # --- Privacy Plot (Right Axis) ---
    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Privacy Risk: Reconstruction PSNR (dB)', color=color, fontsize=12, fontweight='bold')
    # Use dashed line for privacy metric to visually distinguish it
    ax2.plot(x, psnr_scores, marker='s', linestyle='--', color=color, linewidth=2, markersize=8, label='PSNR')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Typically, PSNR above 20-30 dB means recognizable images. Below 10 is noise.
    # Add an academic "Safety threshold" line
    ax2.axhline(y=15, color='gray', linestyle=':', label="Safety Threshold (<15dB)")
    
    # Title and Legend
    plt.title("Privacy-Utility Trade-off under Reconstruction Attack", fontsize=14, fontweight='bold', pad=15)
    
    # Ask matplotlib to gather both lines for a single legend
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right')

    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Plotting] Saved Trade-off plot to {save_path}")

def plot_reconstruction_visuals(
    original_images: List[np.ndarray], 
    reconstructed_images: List[np.ndarray], 
    save_path: str = "outputs/security/visual_leakage.png"
):
    """
    Creates the classic grid plot seen in the DLG paper (Fig 3/4).
    Row 1: Ground Truth
    Row 2: Fully Leaked
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    n = min(len(original_images), 8) # Show max 8 images
    
    fig, axes = plt.subplots(2, n, figsize=(n * 2, 4))
    
    for i in range(n):
        # Ground Truth Row
        ax = axes[0, i] if n > 1 else axes[0]
        ax.imshow(original_images[i].transpose(1, 2, 0)) # assuming C,H,W
        ax.axis('off')
        if i == 0:
            ax.set_title("Ground Truth", fontsize=12, pad=10)
            
        # Reconstructed Row
        ax = axes[1, i] if n > 1 else axes[1]
        ax.imshow(reconstructed_images[i].transpose(1, 2, 0))
        ax.axis('off')
        if i == 0:
            ax.set_title("Fully Leaked", fontsize=12, pad=10)
            
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Plotting] Saved Visual Leakage plot to {save_path}")
