"""MIA dataset and feasibility utilities.

This module inspects server-visible artifacts and reports whether a
membership inference attack is feasible under the strict honest-but-curious
threat model.
"""

from typing import Dict, Any
from ..common_io import load_run_meta, list_rounds, list_clients, load_client_upload
from ..common_types import FeasibilityReport


def analyze_run_feasibility(run_dir: str) -> FeasibilityReport:
    meta = load_run_meta(run_dir)
    sent = meta.get("sent_fields", {})

    # If no sample-level representations, membership inference is limited
    if sent.get("prototypes") and sent.get("class_counts"):
        return FeasibilityReport(
            attack_mode="strict_server_current_protocol",
            status="limited",
            reason="No sample-level representations (only classwise prototypes and counts)",
            available_signals=[k for k, v in sent.items() if v],
            recommended_approach="feasibility_only"
        )

    return FeasibilityReport(
        attack_mode="strict_server_current_protocol",
        status="failed",
        reason="Insufficient server-visible signals for MIA",
        available_signals=[k for k, v in sent.items() if v],
        recommended_approach="none"
    )
"""
MIA dataset builder and utilities.
"""

from typing import List, Dict, Any, Optional
from ..common_types import MIADatasetRow, FeasibilityReport


def assess_mia_feasibility(run_dir: str) -> FeasibilityReport:
    """
    Assess feasibility of MIA based on available server-visible artifacts.
    """
    from ..common_io import load_run_meta, list_rounds, load_client_upload

    meta = load_run_meta(run_dir)
    rounds = list_rounds(run_dir)

    available_signals = []

    # Check what artifacts are available
    for round_num in rounds:
        try:
            server_artifact = load_server_artifact(run_dir, round_num)
            if "global_prototypes" in server_artifact:
                available_signals.append("global_prototypes")
            if "server_reply" in server_artifact:
                available_signals.append("server_reply")
        except:
            pass

        # Check client uploads for additional signals
        try:
            from ..common_io import list_clients
            clients = list_clients(run_dir, round_num)
            if clients:
                upload = load_client_upload(run_dir, round_num, clients[0])
                if "prototype_dict" in upload:
                    available_signals.append("client_prototypes")
                if "class_counts" in upload and upload["class_counts"]:
                    available_signals.append("class_counts")
        except:
            pass

    available_signals = list(set(available_signals))

    # Determine feasibility
    if "client_prototypes" in available_signals:
        # If client prototypes are available, MIA might be possible
        # but we need to check if we have access to sample-level representations
        status = "limited"
        reason = "Client prototypes available, but sample-level representations not visible to server."
        recommended = "Focus on prototype-based membership signals or statistical approaches."
    else:
        status = "limited"
        reason = "No direct sample-level representations available in server-visible artifacts."
        recommended = "MIA not directly feasible with current protocol. Consider statistical attacks on aggregates."

    return FeasibilityReport(
        attack_mode="strict_server_current_protocol",
        status=status,
        reason=reason,
        available_signals=available_signals,
        recommended_approach=recommended,
        metadata={"rounds_analyzed": len(rounds)}
    )


def build_mia_dataset_placeholder(run_dir: str) -> List[MIADatasetRow]:
    """
    Placeholder for MIA dataset building.
    In current protocol, direct MIA dataset cannot be built without sample-level data.
    """
    # This would require access to actual member/non-member samples
    # and their representations, which are not server-visible in the current protocol
    return []


def load_server_artifact(run_dir: str, round_num: int):
    """Helper to load server artifact."""
    from ..common_io import load_server_artifact
    return load_server_artifact(run_dir, round_num)