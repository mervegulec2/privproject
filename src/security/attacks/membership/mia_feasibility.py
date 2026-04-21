from __future__ import annotations

from typing import Any, Dict, List

from src.security.base import BaseAttack
from src.security.attacks.common.types import AttackOutput


class MIAFeasibilityAttack(BaseAttack):
    """
    Feasibility-first MIA under strict honest-but-curious server.

    Reports what the server can (and cannot) do with current server-visible data:
    - class-wise prototypes
    - optional class counts
    - system knowledge: architecture/embedding dim/training procedure/global init
    - auxiliary dataset access (assumed available to attacker)
    """

    def execute(self, model_state: Dict[str, Any], shared_data: Dict[str, Any]) -> Dict[str, Any]:
        clients: List[Dict[str, Any]] = list(shared_data.get("clients") or [])
        if not clients:
            return AttackOutput(status="skipped", reason="no client snapshots").__dict__

        # Inspect one client snapshot for available keys
        sample = clients[0]
        has_protos = bool(sample.get("protos"))
        has_counts = bool(sample.get("counts"))
        has_model_state = model_state is not None

        available = []
        if has_protos:
            available.append("classwise_prototypes")
        if has_counts:
            available.append("class_counts")
        if has_model_state:
            available.append("model_state (system knowledge)")
        available.append("auxiliary_dataset (assumed)")

        # Under strict protocol, record-level membership needs a query representation z(x).
        if not has_protos:
            return AttackOutput(
                status="unsupported",
                reason="no server-visible prototypes; cannot define prototype-mediated membership signals",
                details={"available_signals": available},
                metadata={
                    "required_future_artifacts": ["classwise prototypes (FedProto-like)"],
                },
            ).__dict__

        if not has_model_state:
            # In Flower server we often don't have weights; still can do feasibility and future hooks.
            return AttackOutput(
                status="limited",
                reason="prototypes visible but no model state to embed auxiliary samples (z(x)); feasibility-only",
                details={"available_signals": available},
                metadata={
                    "required_future_artifacts": [
                        "server-side embedding function z(x) (e.g., known init weights or logged model_state)",
                    ]
                },
            ).__dict__

        return AttackOutput(
            status="limited",
            reason="strict server has no true member/non-member ground truth; prototype-mediated scoring possible as a proxy",
            details={"available_signals": available},
            metadata={
                "recommended_attack_mode": "prototype_proxy_scoring",
                "required_future_artifacts": [
                    "member/non-member ground truth or evaluation protocol (proxy-only otherwise)",
                ],
            },
        ).__dict__

