from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from src.security.base import BaseAttack
from src.security.attacks.common.types import AttackOutput
from src.security.attacks.common.utils import client_sent_classes


class TrivialClassPresenceAttack(BaseAttack):
    """
    Trivial CPA (protocol leakage baseline):
    \n  y_hat[c,k] = 1[k in keys(P_c)]\n

    Uses only server-visible uploads: per-client class-wise prototype keys.
    """

    def __init__(self, num_classes: int = 10):
        self.num_classes = num_classes

    def execute(self, model_state: Dict[str, Any], shared_data: Dict[str, Any]) -> Dict[str, Any]:
        clients: List[Dict[str, Any]] = list(shared_data.get("clients") or [])
        if not clients:
            return AttackOutput(status="skipped", reason="no client snapshots").__dict__

        per_client = {}
        for ci in clients:
            cid = ci.get("cid", "?")
            sent = set(client_sent_classes(ci))
            pred = {int(k): int(int(k) in sent) for k in range(self.num_classes)}
            per_client[str(cid)] = {
                "n_sent_classes": int(len(sent)),
                "sent_classes": sorted(list(sent)),
                "pred_presence": pred,
            }

        return AttackOutput(
            status="ok",
            reason="protocol leakage baseline (keys of prototype dict)",
            details={"per_client": per_client, "num_classes": self.num_classes},
        ).__dict__

