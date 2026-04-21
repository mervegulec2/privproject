"""Placeholder for CPA feature extractors and hooks.

Defines feature names and stubs for future learned CPA.
"""

from typing import Dict, Any


def extract_basic_features(row: Dict[str, Any]) -> Dict[str, Any]:
    """Extract minimal features for a CPA row.

    Expected keys in `row`: client_id, class_id, seen_as_key, count, n_sent_classes
    """
    features = {
        "seen_as_key": int(row.get("seen_as_key", 0)),
        "count": row.get("count"),
        "n_sent_classes": row.get("n_sent_classes", 0),
    }
    # slot_norm and similarity placeholders
    features["slot_norm"] = None
    features["mean_similarity"] = None
    features["var_similarity"] = None
    return features
"""
CPA features for learned attacks.
Placeholder for future learned CPA implementations.
"""

# Future: Implement learned CPA using features beyond protocol leakage
# - Prototype distances
# - Count-based features
# - Statistical features

def extract_cpa_features(dataset_row):
    """
    Extract features for learned CPA.
    Currently returns basic features.
    """
    return {
        "seen_as_key": dataset_row.seen_as_key,
        "count": dataset_row.count or 0,
    }


def build_learned_cpa_dataset(run_dir: str):
    """
    Build dataset for learned CPA.
    Placeholder implementation.
    """
    from .cpa_dataset import build_cpa_dataset_extended
    dataset = build_cpa_dataset_extended(run_dir)

    # Extract features
    X = []
    y = []

    for row in dataset:
        features = extract_cpa_features(row)
        X.append(list(features.values()))
        y.append(row.label)

    return X, y