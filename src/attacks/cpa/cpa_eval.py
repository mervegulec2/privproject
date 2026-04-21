"""
CPA evaluation utilities.
"""

from typing import Dict, Any, List
from ..common_types import AttackResult
from ..common_metrics import calculate_attack_metrics, aggregate_metrics
from .cpa_trivial import run_trivial_cpa
from .cpa_dataset import get_cpa_stats
from .cpa_dataset import build_cpa_dataset
from .cpa_features import extract_basic_features


def run_learned_cpa(run_dir: str) -> AttackResult:
    """Train a simple logistic regression on CPA features and evaluate."""
    import numpy as np
    try:
        from sklearn.linear_model import LogisticRegression
    except Exception:
        raise RuntimeError("scikit-learn is required for learned CPA")

    dataset = build_cpa_dataset(run_dir)
    if not dataset:
        return AttackResult(
            attack_name="cpa_learned",
            dataset_size=0,
            metrics={},
            predictions=np.array([]),
            labels=np.array([]),
            scores=np.array([]),
            metadata={"note": "empty dataset"},
        )

    # Build feature matrix
    rows = []
    for row in dataset:
        rdict = {"client_id": row.client_id, "class_id": row.class_id, "seen_as_key": row.seen_as_key, "count": row.count, "n_sent_classes": getattr(row, "n_sent_classes", None)}
        feats = extract_basic_features(rdict)
        rows.append(feats)

    # Convert to numpy array, simple numeric features
    X_list = []
    for f in rows:
        X_list.append([0 if f.get("seen_as_key") is None else int(f.get("seen_as_key")),
                       0 if f.get("count") is None else float(f.get("count")),
                       0 if f.get("n_sent_classes") is None else float(f.get("n_sent_classes"))])
    X = np.array(X_list, dtype=float)
    y = np.array([r.label for r in dataset])

    # Guard: if labels are degenerate, fallback to trivial
    if np.unique(y).size < 2:
        return run_trivial_cpa(run_dir)

    clf = LogisticRegression(max_iter=200)
    clf.fit(X, y)
    probs = clf.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)

    metrics = calculate_attack_metrics(y, probs)

    return AttackResult(
        attack_name="cpa_learned",
        dataset_size=len(dataset),
        metrics=metrics,
        predictions=preds,
        labels=y,
        scores=probs,
        metadata={"model": "logistic_regression", "features": ["seen_as_key","count","n_sent_classes"]},
    )


def evaluate_cpa(run_dir: str, attack_type: str = "trivial") -> AttackResult:
    """
    Evaluate CPA attack on a run directory.
    """
    if attack_type == "trivial":
        return run_trivial_cpa(run_dir)
    else:
        raise ValueError(f"Unknown CPA attack type: {attack_type}")


def evaluate_cpa_multiple_runs(run_dirs: List[str], attack_type: str = "trivial") -> Dict[str, Any]:
    """
    Evaluate CPA across multiple runs and aggregate results.
    """
    results = []
    stats_list = []

    for run_dir in run_dirs:
        result = evaluate_cpa(run_dir, attack_type)
        results.append(result)

        # Get dataset stats
        from .cpa_dataset import build_cpa_dataset
        dataset = build_cpa_dataset(run_dir)
        stats = get_cpa_stats(dataset)
        stats_list.append(stats)

    # Aggregate metrics
    metrics_list = [r.metrics for r in results]
    aggregated_metrics = aggregate_metrics(metrics_list)

    return {
        "attack_type": attack_type,
        "num_runs": len(run_dirs),
        "aggregated_metrics": aggregated_metrics,
        "individual_results": results,
        "dataset_stats": stats_list,
    }


def print_cpa_report(result: Dict[str, Any]) -> None:
    """Print a formatted CPA evaluation report."""
    print("\n" + "="*60)
    print("CPA EVALUATION REPORT")
    print("="*60)
    print(f"Attack Type: {result['attack_type']}")
    print(f"Number of Runs: {result['num_runs']}")
    print()

    print("AGGREGATED METRICS:")
    print("-" * 40)
    from ..common_metrics import print_metrics_summary
    print_metrics_summary(result['aggregated_metrics'])

    print("DATASET STATISTICS:")
    print("-" * 40)
    if result['dataset_stats']:
        stats = result['dataset_stats'][0]  # Show first run stats
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"{key}: {value:.3f}")
            else:
                print(f"{key}: {value}")
    print()