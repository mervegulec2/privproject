"""
Trivial CPA implementation based on protocol leakage.
Directly checks if class keys are present in sent prototypes.
"""

import numpy as np
from typing import List, Dict, Any
from ..common_types import CPADatasetRow, AttackResult
from ..common_metrics import calculate_attack_metrics
from .cpa_dataset import build_cpa_dataset


def run_trivial_cpa(run_dir: str) -> AttackResult:
    """
    Run trivial CPA attack.
    Prediction: class is present if key was sent.
    """
    dataset = build_cpa_dataset(run_dir)

    if len(dataset) == 0:
        return AttackResult(
            attack_name="cpa_trivial",
            dataset_size=0,
            metrics={},
            predictions=np.array([]),
            labels=np.array([]),
            scores=np.array([]),
            metadata={
                "description": "Trivial CPA based on protocol leakage",
                "prediction_rule": "class_present = key_sent",
                "note": "empty dataset"
            }
        )

    # Extract labels and predictions
    labels = np.array([row.label for row in dataset])
    predictions = np.array([row.seen_as_key for row in dataset])
    scores = predictions.astype(float)  # for AUC calculation

    # Calculate metrics
    metrics = calculate_attack_metrics(labels, scores)

    return AttackResult(
        attack_name="cpa_trivial",
        dataset_size=len(dataset),
        metrics=metrics,
        predictions=predictions,
        labels=labels,
        scores=scores,
        metadata={
            "description": "Trivial CPA based on protocol leakage",
            "prediction_rule": "class_present = key_sent"
        }
    )