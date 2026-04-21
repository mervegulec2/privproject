"""Audit script to verify attack code only uses server-visible fields.

Scans `runs/*/round_*/clients/*.pkl` and reports any forbidden keys found in
client uploads (e.g., local embeddings, logits, loss, raw_data, model_state).
"""

import os
import sys
import pickle
import json


FORBIDDEN_KEYS = ["local_embeddings", "logits", "loss", "raw_data", "model_state_dict", "gradients"]


def audit_run(run_dir: str) -> dict:
    report = {"run": run_dir, "violations": []}
    meta_path = os.path.join(run_dir, "meta.json")
    if not os.path.exists(meta_path):
        report["error"] = "no meta.json"
        return report

    for root, _, files in os.walk(run_dir):
        for fn in files:
            if fn.endswith("_upload.pkl"):
                p = os.path.join(root, fn)
                try:
                    with open(p, "rb") as f:
                        data = pickle.load(f)
                    for k in FORBIDDEN_KEYS:
                        if k in data:
                            report["violations"].append({"file": p, "key": k})
                except Exception as e:
                    report.setdefault("errors", []).append({"file": p, "error": str(e)})
    return report


def main(paths):
    results = []
    for p in paths:
        results.append(audit_run(p))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tools/audit_attacks.py <run_dir> [<run_dir> ...]")
        sys.exit(1)
    main(sys.argv[1:])
