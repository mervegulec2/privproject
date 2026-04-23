import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Union, Any

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
    ax1.set_xlabel(r"Defense Strength (Noise Scale $\sigma$)", fontsize=12, fontweight="bold")
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
    comparison_data: List[Dict[str, Any]], 
    save_path: str = "outputs/security/visual_leakage.png"
):
    """
    Creates the classic grid plot seen in the DLG paper (Fig 3/4).
    Row 1: Ground Truth (Centroid)
    Row 2: Reconstructed (Leaked)
    """
    if not comparison_data:
        print("[Plotting] Warning: No data provided for reconstruction plot. Skipping.")
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    n = min(len(comparison_data), 8) # Show max 8 samples
    
    fig, axes = plt.subplots(2, n, figsize=(n * 3, 6))
    
    for i in range(n):
        item = comparison_data[i]
        orig = item["original"]
        recon = item["reconstructed"]
        title = item["title"]

        # Row 0: Original
        ax = axes[0, i] if n > 1 else axes[0]
        ax.imshow(orig)
        ax.set_title(f"Original\n{title}", fontsize=10)
        ax.axis('off')
            
        # Row 1: Reconstructed
        ax = axes[1, i] if n > 1 else axes[1]
        ax.imshow(recon)
        ax.set_title("Reconstructed", fontsize=10)
        ax.axis('off')
            
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Plotting] Saved Visual Leakage plot to {save_path}")

def plot_privacy_utility_tradeoff(
    noise_levels: List[float], 
    accuracies: List[float], 
    psnr_scores: List[float], 
    save_path: str = "outputs/security/tradeoff_plot.png"
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    sns.set_theme(style="whitegrid")
    fig, ax1 = plt.subplots(figsize=(8, 5))
    x = np.arange(len(noise_levels))
    x_labels = [str(nl) if nl > 0 else "Baseline" for nl in noise_levels]
    color = 'tab:blue'
    ax1.set_xlabel('Defense Strength', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', color=color, fontsize=12, fontweight='bold')
    ax1.plot(x, accuracies, marker='o', color=color, label='Accuracy')
    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('PSNR (dB)', color=color, fontsize=12, fontweight='bold')
    ax2.plot(x, psnr_scores, marker='s', linestyle='--', color=color, label='PSNR')
    plt.title("Privacy-Utility Trade-off", fontsize=14, fontweight='bold')
    fig.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_accuracy_curves(data: Union[Dict[str, List[float]], str], save_path: str = "outputs/metrics/accuracy_curve.png"):
    import pandas as pd
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    if isinstance(data, str):
        df = pd.read_csv(data)
        rounds = df["round"].values
        scale = 100 if df["avg_global"].iloc[0] <= 1.0 else 1.0
        plt.plot(rounds, df["avg_global"] * scale, marker='o', label='Global')
        plt.plot(rounds, df["avg_local_proportional"] * scale, marker='s', label='Local Prop')
        plt.plot(rounds, df["avg_local"] * scale, marker='^', label='Local (Weighted)')
    else:
        rounds = range(1, len(data["global"]) + 1)
        plt.plot(rounds, data["global"], marker='o', label='Global')
        if "local" in data:
            plt.plot(rounds, data["local"], marker='^', label='Local (Weighted)')
        if "local_prop" in data:
            plt.plot(rounds, data["local_prop"], marker='s', label='Local Prop')
    plt.legend()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_mia_distribution(member_scores: List[float], non_member_scores: List[float], save_path: str = "outputs/security/eval_results/mia_distribution.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    sns.set_theme(style="white")
    plt.figure(figsize=(10, 6))
    sns.histplot(member_scores, label="Members", color="blue", kde=True, stat="density", alpha=0.5)
    sns.histplot(non_member_scores, label="Non-Members", color="red", kde=True, stat="density", alpha=0.5)
    plt.title("MIA Score Distribution")
    plt.legend()
    plt.savefig(save_path, dpi=300)
    plt.close()
