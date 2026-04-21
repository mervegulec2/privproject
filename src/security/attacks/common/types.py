from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AttackOutput:
    """Lightweight, JSON-serializable-ish attack output."""

    status: str  # e.g. "ok", "limited", "unsupported", "skipped", "error"
    reason: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)  # paths, if any
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

