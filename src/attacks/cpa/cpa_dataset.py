"""
CPA dataset builder and utilities.
"""

from typing import List, Dict, Any
from ..common_types import CPADatasetRow
from ..common_io import load_run_meta, list_rounds, list_clients, load_client_upload


def build_cpa_dataset(run_dir: str) -> List[CPADatasetRow]:
    """Build CPA dataset from run artifacts.

    Each row corresponds to a (client_id, class_id) pair. Only server-visible
    fields (as determined by `meta.json.sent_fields`) are used.
    """
    meta = load_run_meta(run_dir)
    n_classes = meta.get("n_classes", 10)
    rounds = list_rounds(run_dir)

    dataset: List[CPADatasetRow] = []

    for round_num in rounds:
        clients = list_clients(run_dir, round_num)
        for client_id in clients:
            upload = load_client_upload(run_dir, round_num, client_id)
            sent_classes = set(upload.get("sent_classes") or [])
            class_counts = upload.get("class_counts") or {}

            for class_id in range(n_classes):
                label = 1 if class_id in sent_classes else 0
                seen_as_key = 1 if class_id in sent_classes else 0
                count = class_counts.get(class_id)
                row = CPADatasetRow(
                    client_id=client_id,
                    class_id=class_id,
                    label=label,
                    seen_as_key=seen_as_key,
                    count=count,
                )
                dataset.append(row)

    return dataset


def build_cpa_dataset_extended(run_dir: str) -> List[CPADatasetRow]:
    """
    Build extended CPA dataset with additional features.
    Currently same as basic, but extensible for future features.
    """
    from .cpa_trivial import build_cpa_dataset
    return build_cpa_dataset(run_dir)


def get_cpa_stats(dataset: List[CPADatasetRow]) -> Dict[str, Any]:
    """Get statistics about the CPA dataset."""
    total_samples = len(dataset)
    positive_samples = sum(1 for row in dataset if row.label == 1)
    negative_samples = total_samples - positive_samples

    clients = set(row.client_id for row in dataset)
    classes = set(row.class_id for row in dataset)

    return {
        "total_samples": total_samples,
        "positive_samples": positive_samples,
        "negative_samples": negative_samples,
        "positive_rate": positive_samples / total_samples,
        "num_clients": len(clients),
        "num_classes": len(classes),
        "samples_per_client": total_samples / len(clients),
        "samples_per_class": total_samples / len(classes),
    }


def filter_cpa_dataset(dataset: List[CPADatasetRow], client_ids: List[int] = None,
                      class_ids: List[int] = None) -> List[CPADatasetRow]:
    """Filter CPA dataset by client or class IDs."""
    filtered = dataset

    if client_ids is not None:
        client_ids = set(client_ids)
        filtered = [row for row in filtered if row.client_id in client_ids]

    if class_ids is not None:
        class_ids = set(class_ids)
        filtered = [row for row in filtered if row.class_id in class_ids]

    return filtered