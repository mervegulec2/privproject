"""Define feature modes for MIA under current and future protocols.

Modes:
- prototype_only
- counts_only
- proto_plus_counts
- mean_only (future)
- mean_var (future)
"""

AVAILABLE_MODES = [
    "prototype_only",
    "counts_only",
    "proto_plus_counts",
    "mean_only",
    "mean_var",
    "mean_cov",
]


def validate_mode(mode: str) -> bool:
    return mode in AVAILABLE_MODES
"""
MIA features for advanced attacks.
Placeholder for future mean/variance/covariance based attacks.
"""

# Future: Implement features when mean/variance artifacts become available
# - Mahalanobis distance features
# - Statistical membership tests
# - Distribution-based scores

def extract_mia_features(sample_repr: np.ndarray, client_stats: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract MIA features from sample representation and client statistics.
    Currently placeholder for future statistical features.
    """
    features = {}

    # Placeholder for mean-based features
    if "mean_dict" in client_stats:
        # Mahalanobis distance would go here
        features["mahalanobis_distance"] = 0.0  # placeholder

    # Placeholder for variance-based features
    if "var_dict" in client_stats:
        features["variance_score"] = 0.0  # placeholder

    return features


def score_membership(sample_repr: np.ndarray, client_stats: Dict[str, Any], mode: str = "mean_var") -> float:
    """
    Score membership probability using statistical features.
    """
    if mode == "mean_var" and "mean_dict" in client_stats and "var_dict" in client_stats:
        # Implement Mahalanobis distance
        return 0.5  # placeholder
    elif mode == "mean_cov" and "mean_dict" in client_stats and "cov_dict" in client_stats:
        # Implement full Mahalanobis
        return 0.5  # placeholder
    else:
        # Fallback to neutral score
        return 0.5