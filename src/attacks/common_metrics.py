"""
Common metrics utilities for attack evaluation.
Provides ROC-AUC, PR-AUC, TPR@FPR calculations.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from typing import Tuple, Dict, List


def calculate_roc_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Calculate ROC-AUC score."""
    # Handle degenerate cases where ROC AUC cannot be computed
    if y_true.size == 0:
        return None
    unique = np.unique(y_true)
    if unique.size < 2:
        return None
    return roc_auc_score(y_true, y_scores)


def calculate_pr_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Calculate Precision-Recall AUC score."""
    if y_true.size == 0:
        return None
    unique = np.unique(y_true)
    if unique.size < 2:
        return None
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    return auc(recall, precision)


def calculate_tpr_at_fpr(y_true: np.ndarray, y_scores: np.ndarray, target_fpr: float = 0.01) -> float:
    """Calculate TPR at a specific FPR threshold."""
    from sklearn.metrics import roc_curve
    if y_true.size == 0:
        return None
    unique = np.unique(y_true)
    if unique.size < 2:
        return None
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    # Find the TPR at the target FPR
    idx = np.argmin(np.abs(fpr - target_fpr))
    return float(tpr[idx])


def calculate_attack_metrics(y_true: np.ndarray, y_scores: np.ndarray) -> Dict[str, float]:
    """Calculate all standard attack metrics."""
    metrics = {
        "roc_auc": calculate_roc_auc(y_true, y_scores),
        "pr_auc": calculate_pr_auc(y_true, y_scores),
        "tpr_at_1_fpr": calculate_tpr_at_fpr(y_true, y_scores, 0.01),
        "tpr_at_0_1_fpr": calculate_tpr_at_fpr(y_true, y_scores, 0.001),
    }
    return metrics


def aggregate_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Aggregate metrics across multiple runs/seeds."""
    if not metrics_list:
        return {}

    keys = metrics_list[0].keys()
    aggregated = {}

    for key in keys:
        values = [m[key] for m in metrics_list]
        aggregated[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    return aggregated


def print_metrics_summary(metrics: Dict[str, Dict[str, float]]) -> None:
    """Print a formatted summary of aggregated metrics."""
    print("\nAttack Metrics Summary:")
    print("-" * 60)
    for metric_name, stats in metrics.items():
        print(f"{metric_name.upper()}: ")
        if not isinstance(stats, dict) or not stats:
            print("  No data")
            continue
        mean = stats.get("mean")
        std = stats.get("std")
        mn = stats.get("min")
        mx = stats.get("max")
        mean_s = "None" if mean is None else f"{mean:.4f}"
        std_s = "None" if std is None else f"{std:.4f}"
        min_s = "None" if mn is None else f"{mn:.4f}"
        max_s = "None" if mx is None else f"{mx:.4f}"
        print(f"  mean: {mean_s} | std: {std_s} | min: {min_s} | max: {max_s}")
        print()