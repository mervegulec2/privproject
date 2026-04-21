from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

import numpy as np


def safe_cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + eps
    return float(np.dot(a, b) / denom)


def safe_l2_neg(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    return -float(np.linalg.norm(a - b))


def client_sent_classes(client_info: Dict[str, Any]) -> List[int]:
    protos = client_info.get("protos") or {}
    try:
        return sorted(int(k) for k in protos.keys())
    except Exception:
        # if keys are not ints
        return sorted(list(protos.keys()))

