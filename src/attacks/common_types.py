"""
Common type definitions for attack framework.
"""

from typing import Dict, List, Any, Optional, NamedTuple
from dataclasses import dataclass
import numpy as np


@dataclass
class AttackResult:
    """Result of a single attack evaluation."""
    attack_name: str
    dataset_size: int
    metrics: Dict[str, float]
    predictions: np.ndarray
    labels: np.ndarray
    scores: np.ndarray
    metadata: Dict[str, Any]


@dataclass
class FeasibilityReport:
    """Report on attack feasibility."""
    attack_mode: str
    status: str  # "success", "limited", "failed"
    reason: str
    available_signals: List[str]
    recommended_approach: str
    metadata: Dict[str, Any]


@dataclass
class ClientData:
    """Data from a single client."""
    client_id: int
    round_num: int
    sent_classes: List[int]
    prototype_dict: Dict[int, np.ndarray]
    class_counts: Optional[Dict[int, int]]
    proto_dim: int
    n_sent_classes: int


@dataclass
class ServerData:
    """Data from server aggregation."""
    global_prototypes: Dict[int, np.ndarray]
    server_reply: Any
    global_metrics: Dict[str, float]


class CPADatasetRow(NamedTuple):
    """Row in CPA dataset."""
    client_id: int
    class_id: int
    label: int  # 1 if class is present in client's data, 0 otherwise
    seen_as_key: int  # 1 if class key was sent, 0 otherwise
    count: Optional[int]  # number of samples if available


class MIADatasetRow(NamedTuple):
    """Row in MIA dataset."""
    client_id: int
    sample_id: int
    label: int  # 1 if member, 0 if non-member
    score: float  # membership score
    features: Dict[str, Any]  # additional features