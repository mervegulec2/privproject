"""I/O helpers for attack framework.

Only expose server-visible artifacts. Functions here read
`runs/{run_id}/meta.json`, per-round `server_artifact.pkl`, and
`clients/client_{cid}_upload.pkl` files, but only return fields
that `meta.json` marks as sent (honest-but-curious threat model).
"""

import os
import json
import pickle
from typing import List, Dict, Any, Optional

from .common_types import ClientData, ServerData


def load_run_meta(run_dir: str) -> Dict[str, Any]:
    path = os.path.join(run_dir, "meta.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"meta.json not found for run: {run_dir}")
    with open(path, "r") as f:
        meta = json.load(f)
    # Ensure sent_fields exists
    meta.setdefault("sent_fields", {})
    return meta


def list_rounds(run_dir: str) -> List[int]:
    rounds = []
    for name in os.listdir(run_dir):
        if name.startswith("round_"):
            try:
                rounds.append(int(name.split("_")[1]))
            except Exception:
                continue
    return sorted(rounds)


def list_clients(run_dir: str, round_num: int) -> List[int]:
    clients_dir = os.path.join(run_dir, f"round_{round_num}", "clients")
    if not os.path.exists(clients_dir):
        return []
    clients = []
    for fn in os.listdir(clients_dir):
        if fn.startswith("client_") and fn.endswith("_upload.pkl"):
            parts = fn.split("_")
            try:
                cid = int(parts[1])
                clients.append(cid)
            except Exception:
                continue
    return sorted(clients)


def load_client_upload(run_dir: str, round_num: int, client_id: int) -> Dict[str, Any]:
    path = os.path.join(run_dir, f"round_{round_num}", "clients", f"client_{client_id}_upload.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "rb") as f:
        data = pickle.load(f)

    # Only keep server-visible fields according to meta
    meta = load_run_meta(run_dir)
    sent = meta.get("sent_fields", {})

    safe = {
        "client_id": data.get("client_id"),
        "round": data.get("round"),
        "sent_artifact_type": data.get("sent_artifact_type"),
        # classes and counts are explicitly server-visible in our protocol
        "sent_classes": data.get("sent_classes") if sent.get("prototypes", False) else None,
        "prototype_dict": data.get("prototype_dict") if sent.get("prototypes", False) else None,
        "class_counts": data.get("class_counts") if sent.get("class_counts", False) else None,
        "proto_dim": data.get("proto_dim") if sent.get("prototypes", False) else None,
        "n_sent_classes": data.get("n_sent_classes"),
    }
    return safe


def load_server_artifact(run_dir: str, round_num: int) -> ServerData:
    path = os.path.join(run_dir, f"round_{round_num}", "server_artifact.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "rb") as f:
        data = pickle.load(f)

    # Only return server-visible aggregate fields
    safe = ServerData(
        global_prototypes=data.get("global_prototypes"),
        server_reply=data.get("server_reply"),
        global_metrics=data.get("global_metrics", {}),
    )
    return safe


def save_attack_results(run_dir: str, result: Any, name: str) -> None:
    """Save attack results to `run_dir/attacks/{name}.json` and a CSV summary.

    Note: keep outputs separate from run artifacts.
    """
    import json
    import csv

    out_dir = os.path.join(run_dir, "attacks")
    os.makedirs(out_dir, exist_ok=True)

    json_path = os.path.join(out_dir, f"{name}.json")
    with open(json_path, "w") as f:
        json.dump(result_to_json(result), f, indent=2)

    # If result contains per-sample predictions, write CSV
    csv_path = os.path.join(out_dir, f"{name}.csv")
    if isinstance(result, dict) and "individual_results" in result:
        # try to flatten first result
        first = result["individual_results"][0]
        try:
            preds = getattr(first, "predictions", None)
            labels = getattr(first, "labels", None)
            scores = getattr(first, "scores", None)
            if preds is not None and labels is not None:
                with open(csv_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["prediction", "label", "score"])
                    for p, l, s in zip(preds, labels, scores if scores is not None else [None]*len(preds)):
                        writer.writerow([int(p), int(l), float(s) if s is not None else ""])
        except Exception:
            pass


def result_to_json(result: Any) -> Any:
    """Helper to convert AttackResult or dict to JSON-serializable structure."""
    if hasattr(result, "__dict__"):
        data = result.__dict__
    else:
        data = result
    # sanitize numpy arrays
    def _sanitize(o):
        import numpy as _np
        if _np is not None and isinstance(o, _np.ndarray):
            return o.tolist()
        if isinstance(o, dict):
            return {k: _sanitize(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_sanitize(x) for x in o]
        return o

    return _sanitize(data)
"""
Common I/O utilities for attack framework.
Handles loading runs, parsing artifacts, and logging utilities.
"""

import os
import pickle
import json
from typing import Dict, List, Any, Optional
from pathlib import Path


def load_run_meta(run_dir: str) -> Dict[str, Any]:
    """Load meta.json from a run directory."""
    meta_path = os.path.join(run_dir, "meta.json")
    with open(meta_path, "r") as f:
        return json.load(f)


def load_client_upload(run_dir: str, round_num: int, client_id: int) -> Dict[str, Any]:
    """Load client upload pickle for a specific round and client."""
    upload_path = os.path.join(run_dir, f"round_{round_num}", "clients", f"client_{client_id}_upload.pkl")
    with open(upload_path, "rb") as f:
        return pickle.load(f)


def load_server_artifact(run_dir: str, round_num: int) -> Dict[str, Any]:
    """Load server artifact pickle for a specific round."""
    artifact_path = os.path.join(run_dir, f"round_{round_num}", "server_artifact.pkl")
    with open(artifact_path, "rb") as f:
        return pickle.load(f)


def list_rounds(run_dir: str) -> List[int]:
    """List available round numbers in a run directory."""
    rounds = []
    for item in os.listdir(run_dir):
        if item.startswith("round_") and os.path.isdir(os.path.join(run_dir, item)):
            round_num = int(item.split("_")[1])
            rounds.append(round_num)
    return sorted(rounds)


def list_clients(run_dir: str, round_num: int) -> List[int]:
    """List available client IDs for a specific round."""
    clients_dir = os.path.join(run_dir, f"round_{round_num}", "clients")
    clients = []
    for item in os.listdir(clients_dir):
        if item.startswith("client_") and item.endswith("_upload.pkl"):
            client_id = int(item.split("_")[1])
            clients.append(client_id)
    return sorted(clients)


def save_attack_results(run_dir: str, results: Dict[str, Any], attack_name: str) -> None:
    """Save attack results to the run directory."""
    results_dir = os.path.join(run_dir, "attacks")
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f"{attack_name}_results.pkl")
    with open(results_path, "wb") as f:
        pickle.dump(results, f)


def load_attack_results(run_dir: str, attack_name: str) -> Optional[Dict[str, Any]]:
    """Load attack results from the run directory."""
    results_path = os.path.join(run_dir, "attacks", f"{attack_name}_results.pkl")
    if os.path.exists(results_path):
        with open(results_path, "rb") as f:
            return pickle.load(f)
    return None