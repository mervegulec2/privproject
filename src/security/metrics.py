import numpy as np
import torch
import scipy.stats
from sklearn.metrics import roc_auc_score, mean_squared_error, precision_recall_fscore_support, roc_curve
from typing import Dict, Any, Union, List

# =============================================================================
# TIER 1: GENERAL STATISTICAL & INFORMATION-THEORETIC METRICS
# (Used to quantify the fundamental Privacy vs. Information Trade-off)
# =============================================================================

def calculate_statistical_leakage(original: np.ndarray, modified: np.ndarray) -> Dict[str, float]:
    """
    General measures of information leakage and statistical distortion.
    Applicable to any defense (Noise, Quantization, etc.)
    """
    o_flat = original.flatten()
    m_flat = modified.flatten()
    
    # 1. KL-Divergence (Measure of information gain/loss)
    p = np.histogram(o_flat, bins=50, density=True)[0] + 1e-10
    q = np.histogram(m_flat, bins=50, density=True)[0] + 1e-10
    kl_div = scipy.stats.entropy(p, q)
    
    # 2. Pearson Correlation (Linear relationship leakage)
    correlation = np.corrcoef(o_flat, m_flat)[0, 1]
    
    return {
        "kl_divergence": float(kl_div),
        "correlation_leakage": float(correlation)
    }

# =============================================================================
# TIER 2: UTILITY & COST METRICS
# =============================================================================

def calculate_general_utility(baseline_acc: float, current_acc: float) -> Dict[str, float]:
    """Measures how much 'Utility' remains for the FL tasks."""
    return {
        "accuracy_tax": float(baseline_acc - current_acc),
        "utility_retention": float(current_acc / (baseline_acc + 1e-8))
    }

# =============================================================================
# TIER 3: SPECIALIZED RECONSTRUCTION METRICS
# =============================================================================

def get_reconstruction_fidelity(original: np.ndarray, reconstructed: np.ndarray) -> Dict[str, float]:
    """Structural and visual leakage metrics."""
    if isinstance(original, torch.Tensor): original = original.detach().cpu().numpy()
    if isinstance(reconstructed, torch.Tensor): reconstructed = reconstructed.detach().cpu().numpy()
    
    o_flat = original.flatten()
    r_flat = reconstructed.flatten()
    
    mse = mean_squared_error(o_flat, r_flat)
    psnr = 100.0 if mse == 0 else 20 * np.log10(1.0 / np.sqrt(mse))
    
    # SSIM Approximation
    mu_o, mu_r = o_flat.mean(), r_flat.mean()
    var_o, var_r = o_flat.var(), r_flat.var()
    cov = np.cov(o_flat, r_flat)[0, 1]
    c1, c2 = 0.0001, 0.0009
    ssim = ((2*mu_o*mu_r + c1)*(2*cov + c2)) / ((mu_o**2 + mu_r**2 + c1)*(var_o + var_r + c2))
    
    return {
        "psnr": float(psnr),
        "ssim": float(ssim),
        "cosine_sim": float(np.dot(o_flat, r_flat) / (np.linalg.norm(o_flat)*np.linalg.norm(r_flat) + 1e-8))
    }

# =============================================================================
# TIER 4: SPECIALIZED INFERENCE METRICS (MIA)
# =============================================================================

def get_inference_leakage(member_scores: List[float], non_member_scores: List[float]) -> Dict[str, float]:
    """Scientific MIA metrics: AUC-ROC, Advantage, and Confidence Gap."""
    if not member_scores or not non_member_scores:
        return {"auc_roc": 0.5, "attacker_advantage": 0.0}
        
    y_true = [1] * len(member_scores) + [0] * len(non_member_scores)
    y_scores = member_scores + non_member_scores
    
    auc = roc_auc_score(y_true, y_scores)
    advantage = max(0, 2 * auc - 1)
    
    return {
        "auc_roc": float(auc),
        "attacker_advantage": float(advantage),
        "confidence_gap": float(np.mean(member_scores) - np.mean(non_member_scores))
    }

def get_tpr_at_fpr(member_scores: List[float], non_member_scores: List[float], fpr_target: float = 0.01) -> float:
    """True Positive Rate at limited False Positive Rate (Scientific Security)."""
    if not member_scores or not non_member_scores: return 0.0
    y_true = [1] * len(member_scores) + [0] * len(non_member_scores)
    y_scores = member_scores + non_member_scores
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    idx = np.argmin(np.abs(fpr - fpr_target))
    return float(tpr[idx])

# =============================================================================
# TIER 5: CATEGORICAL ATTRIBUTION METRICS (Class Presence Attack)
# =============================================================================

def get_categorical_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """Precision, Recall, and F1 for detecting class existence."""
    if not any(y_true) and not any(y_pred):
         return {"f1_score": 1.0, "precision": 1.0, "recall": 1.0}
         
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    
    return {
        "f1_score": float(f1),
        "precision": float(prec),
        "recall": float(rec)
    }
