import numpy as np
import torch
import scipy.stats
from sklearn.metrics import roc_auc_score, mean_squared_error
from typing import Dict, Any, Union

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
    # We use a small epsilon to avoid division by zero
    p = np.histogram(o_flat, bins=50, density=True)[0] + 1e-10
    q = np.histogram(m_flat, bins=50, density=True)[0] + 1e-10
    kl_div = scipy.stats.entropy(p, q)
    
    # 2. Pearson Correlation (Linear relationship leakage)
    correlation = np.corrcoef(o_flat, m_flat)[0, 1]
    
    # 3. L2 Distortion (Physical distance)
    distortion = np.linalg.norm(o_flat - m_flat)
    
    return {
        "kl_divergence": float(kl_div),
        "correlation_leakage": float(correlation),
        "data_distortion": float(distortion)
    }

# =============================================================================
# TIER 2: UTILITY & COST METRICS
# (Used to quantify the 'Price' of Privacy)
# =============================================================================

def calculate_general_utility(baseline_acc: float, current_acc: float) -> Dict[str, float]:
    """Measures how much 'Utility' remains for the FL tasks."""
    return {
        "accuracy_tax": float(baseline_acc - current_acc),
        "utility_retention": float(current_acc / (baseline_acc + 1e-8)),
        "is_converged": bool(current_acc > 0.1) # General baseline check
    }

# =============================================================================
# TIER 3: SPECIALIZED RECONSTRUCTION METRICS
# (Used only for Reconstruction/Inversion Attacks)
# =============================================================================

def get_reconstruction_fidelity(original: np.ndarray, reconstructed: np.ndarray) -> Dict[str, float]:
    """Structural and visual leakage metrics."""
    # Convert to numpy
    if isinstance(original, torch.Tensor): original = original.detach().cpu().numpy()
    if isinstance(reconstructed, torch.Tensor): reconstructed = reconstructed.detach().cpu().numpy()
    
    o_flat = original.flatten()
    r_flat = reconstructed.flatten()
    
    # PSNR
    mse = mean_squared_error(o_flat, r_flat)
    psnr = 100.0 if mse == 0 else 20 * np.log10(1.0 / np.sqrt(mse))
    
    # SSIM (Structural Similarity)
    mu_o, mu_r = o_flat.mean(), r_flat.mean()
    var_o, var_r = o_flat.var(), r_flat.var()
    cov = np.cov(o_flat, r_flat)[0, 1]
    c1, c2 = 0.0001, 0.0009
    ssim = ((2*mu_o*mu_r + c1)*(2*cov + c2)) / ((mu_o**2 + mu_r**2 + c1)*(var_o + var_r + c2))
    
    return {
        "mse": float(mse),
        "psnr": float(psnr),
        "ssim": float(ssim),
        "cosine_sim": float(np.dot(o_flat, r_flat) / (np.linalg.norm(o_flat)*np.linalg.norm(r_flat) + 1e-8))
    }

# =============================================================================
# TIER 4: SPECIALIZED INFERENCE METRICS
# (Used only for Membership/Property Inference Attacks)
# =============================================================================

def get_inference_leakage(y_true: np.ndarray, y_scores: np.ndarray) -> Dict[str, float]:
    """Binary identification and membership metrics."""
    auc = roc_auc_score(y_true, y_scores)
    # Attacker advantage (0 to 1)
    advantage = max(0, 2 * auc - 1)
    
    return {
        "auc_roc": float(auc),
        "attacker_advantage": float(advantage)
    }
