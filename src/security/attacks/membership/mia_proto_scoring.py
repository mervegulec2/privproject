from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.security.base import BaseAttack
from src.security.attacks.common.types import AttackOutput
from src.security.attacks.common.utils import safe_cosine, safe_l2_neg
from src.data_utils import load_cifar10, Cifar10Config
from src.models import ResNet18Cifar


class PrototypeMIAAttack(BaseAttack):
    """
    Prototype-mediated membership scoring (proxy baseline).

    Threat model:
    - server sees class-wise prototypes (per client)
    - server knows architecture and may know/init weights (model_state)
    - server has auxiliary data (CIFAR-10 test set)

    This is NOT a strict, ground-truth record-level MIA unless you define
    a membership evaluation protocol. We report scores only.
    """

    def __init__(self, num_classes: int = 10, max_aux_samples: int = 512, scorer: str = "cosine"):
        self.num_classes = num_classes
        self.max_aux_samples = max_aux_samples
        self.scorer = scorer  # "cosine" | "l2"

    def _embed_aux(self, model_state: Dict[str, Any], device: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return (embeddings [N,D], labels [N])."""
        model = ResNet18Cifar(num_classes=self.num_classes)
        model.load_state_dict(model_state, strict=True)
        model.to(device)
        model.eval()

        _, test_ds = load_cifar10(Cifar10Config(root="data"))
        n = min(self.max_aux_samples, len(test_ds))
        loader = DataLoader(torch.utils.data.Subset(test_ds, list(range(n))), batch_size=128, shuffle=False)

        embs = []
        ys = []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                _, e = model(x)
                embs.append(e.detach().cpu().numpy())
                ys.append(y.numpy())

        return np.concatenate(embs, axis=0), np.concatenate(ys, axis=0)

    def execute(self, model_state: Dict[str, Any], shared_data: Dict[str, Any]) -> Dict[str, Any]:
        # Strict default: unless explicitly opted-in, do NOT use any model_state.
        # This keeps the strict honest-but-curious threat model intact.
        allow_model_state = bool(shared_data.get("log_model_state", False))
        if not allow_model_state:
            return AttackOutput(
                status="unsupported",
                reason="strict_server_valid: mia_proto requires server-visible embedding function z(x); model_state logging is disabled",
            ).__dict__

        clients: List[Dict[str, Any]] = list(shared_data.get("clients") or [])
        if not clients:
            return AttackOutput(status="skipped", reason="no client snapshots").__dict__
        if model_state is None:
            return AttackOutput(
                status="unsupported",
                reason="strict_server_valid: missing model_state; cannot compute auxiliary embeddings z(x) on server",
            ).__dict__

        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            aux_embs, aux_labels = self._embed_aux(model_state, device=device)
        except Exception as e:
            return AttackOutput(status="error", reason=f"failed to embed auxiliary data: {e}").__dict__

        # scoring function
        if self.scorer == "l2":
            score_fn = safe_l2_neg
        else:
            score_fn = safe_cosine

        per_client = {}
        for ci in clients:
            cid = ci.get("cid", "?")
            protos: Dict[int, np.ndarray] = ci.get("protos") or {}
            if not protos:
                per_client[str(cid)] = {"status": "skipped", "reason": "no prototypes"}
                continue

            # compute score for each auxiliary sample w.r.t its label prototype (if exists)
            scores = []
            used = 0
            for z, y in zip(aux_embs, aux_labels):
                y_int = int(y)
                if y_int not in protos:
                    continue
                s = score_fn(z, protos[y_int])
                scores.append(float(s))
                used += 1

            per_client[str(cid)] = {
                "status": "ok",
                "scorer": self.scorer,
                "n_aux_used": int(used),
                "score_mean": float(np.mean(scores)) if scores else None,
                "score_std": float(np.std(scores)) if scores else None,
                "scores_preview": scores[:20],
            }

        return AttackOutput(
            status="limited",
            reason="proxy membership scoring over auxiliary data (no membership ground truth)",
            details={"per_client": per_client, "max_aux_samples": self.max_aux_samples},
        ).__dict__

