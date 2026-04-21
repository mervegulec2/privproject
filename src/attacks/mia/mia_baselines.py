"""MIA baseline helpers: feasibility checks and scorer interface.

Provides a scorer interface for mean/variance attacks and a small demo
simulation function to test the scorer on synthetic embeddings.
"""

from typing import Any, Dict, List
import numpy as np
from ..common_types import AttackResult


def membership_scorer_mean_var(sample_repr: Any, client_stats: Dict[str, Any], eps: float = 1e-6) -> float:
    """Compute a Mahalanobis-like membership score using diagonal variance.

    score = - sum_d ((z_d - mu_d)^2 / (var_d + eps)). Higher (less negative)
    indicates closer to client distribution -> more likely member.
    """
    z = np.asarray(sample_repr, dtype=float)
    mu = np.asarray(client_stats.get("mean"), dtype=float)
    var = np.asarray(client_stats.get("var"), dtype=float)
    if z.shape != mu.shape or var.shape != mu.shape:
        raise ValueError("Shape mismatch between sample_repr and client_stats mean/var")
    denom = var + eps
    score = -float(np.sum(((z - mu) ** 2) / denom))
    return score


def simulate_mia_demo(n_clients: int = 2, dim: int = 16, n_samples_per_client: int = 50) -> Dict[str, Any]:
    """Simulate simple membership scoring experiment.

    Returns ROC-AUC for a Mahalanobis-like scorer where client means/vars
    are known (simulating server-visible stats).
    """
    from sklearn.metrics import roc_auc_score

    # Generate client stats
    client_stats = {}
    samples = []
    labels = []

    for c in range(n_clients):
        mu = np.random.randn(dim) * 2.0
        var = np.abs(np.random.rand(dim)) + 0.5
        client_stats[c] = {"mean": mu, "var": var}

        # member samples
        for i in range(n_samples_per_client):
            z = mu + np.random.randn(dim) * np.sqrt(var)
            samples.append((z, c, 1))

    # add non-member samples (from global noise)
    for i in range(n_clients * n_samples_per_client):
        z = np.random.randn(dim) * 3.0
        samples.append((z, None, 0))

    # compute scores: for each sample, score against its claimed client (choose client 0)
    scores = []
    y = []
    for z, c, label in samples:
        # pick a target client to test membership against (client 0)
        stats = client_stats[0]
        s = membership_scorer_mean_var(z, stats)
        scores.append(s)
        y.append(label)

    try:
        auc = roc_auc_score(y, scores)
    except Exception:
        auc = None

    return {"roc_auc": auc, "n_samples": len(y)}


def feasibility_from_signals(signals: Dict[str, bool]) -> Dict[str, Any]:
    return {
        "has_prototypes": signals.get("prototypes", False),
        "has_counts": signals.get("class_counts", False),
        "membership_possible": signals.get("prototypes", False) and signals.get("class_counts", False),
    }