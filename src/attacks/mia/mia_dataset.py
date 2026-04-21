"""MIA feasibility utilities (strict server-visible protocol).

Under the strict honest-but-curious server model, record-level MIA is a
feasibility question first: without any server-visible sample-level
representations, classic per-record membership classifiers are not directly
constructible.
"""

from typing import Dict, Any, List
from ..common_io import load_run_meta, list_rounds, list_clients, load_client_upload
from ..common_types import FeasibilityReport


def analyze_run_feasibility(run_dir: str) -> FeasibilityReport:
    meta = load_run_meta(run_dir)
    sent = meta.get("sent_fields", {})

    available_signals = [k for k, v in sent.items() if v]

    # Current strict protocol: classwise prototypes (+ optional counts) only.
    # No record-level / sample-level representation is visible to server.
    if sent.get("prototypes", False):
        required_future: List[str] = [
            "per-sample representations (or per-sample summaries)",
            "or (mean,var) per class + a server-visible query representation z",
        ]
        if sent.get("class_counts", False):
            reason = "Only classwise prototypes and counts are server-visible; no sample-level representation."
        else:
            reason = "Only classwise prototypes are server-visible; no counts and no sample-level representation."

        return FeasibilityReport(
            attack_mode="strict_server_current_protocol",
            status="limited",
            reason=reason,
            available_signals=available_signals,
            recommended_approach="feasibility_only",
            required_future_artifacts=required_future,
            metadata={"protocol": "strict_server_visible"},
        )

    return FeasibilityReport(
        attack_mode="strict_server_current_protocol",
        status="failed",
        reason="No server-visible prototypes (and no sample-level representation) available.",
        available_signals=available_signals,
        recommended_approach="none",
        required_future_artifacts=[
            "client upload of classwise prototypes (FedProto-like) and/or counts",
            "or sample-level representations (not recommended under strict server model)",
        ],
        metadata={"protocol": "strict_server_visible"},
    )