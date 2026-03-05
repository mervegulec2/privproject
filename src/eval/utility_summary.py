"""
FAZ 1 utility evaluation: read client_accuracy.csv and compute/save summaries.
Produces: mean accuracy, std, min, max, per-client table; saves to text + JSON.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict


def load_client_accuracy(csv_path: str) -> list[dict]:
    """Load client_accuracy.csv into list of row dicts."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            row["local_test_accuracy"] = float(row["local_test_accuracy"])
            row["num_train_samples"] = int(row["num_train_samples"])
            row["num_proto_classes"] = int(row["num_proto_classes"])
            row["client_id"] = int(row["client_id"])
            rows.append(row)
    return rows


def summarize(rows: list[dict]) -> dict:
    """Compute summary stats for a list of client rows (same alpha/seed)."""
    if not rows:
        return {}
    accs = [r["local_test_accuracy"] for r in rows]
    n = len(accs)
    mean_acc = sum(accs) / n
    variance = sum((a - mean_acc) ** 2 for a in accs) / n if n > 0 else 0
    std_acc = variance ** 0.5
    return {
        "num_clients": n,
        "mean_accuracy": round(mean_acc, 6),
        "std_accuracy": round(std_acc, 6),
        "min_accuracy": round(min(accs), 6),
        "max_accuracy": round(max(accs), 6),
        "per_client_accuracy": [round(a, 6) for a in accs],
        "per_client": [
            {
                "client_id": r["client_id"],
                "local_test_accuracy": round(r["local_test_accuracy"], 6),
                "num_train_samples": r["num_train_samples"],
                "num_proto_classes": r["num_proto_classes"],
            }
            for r in sorted(rows, key=lambda x: x["client_id"])
        ],
    }


def main():
    ap = argparse.ArgumentParser(description="Summarize FL utility from client_accuracy.csv")
    ap.add_argument(
        "--csv",
        type=str,
        default="outputs/metrics/client_accuracy.csv",
        help="Path to client_accuracy.csv",
    )
    ap.add_argument(
        "--output-dir",
        type=str,
        default="outputs/metrics",
        help="Directory for summary output files",
    )
    ap.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Filter by alpha (e.g. 0.1). If not set, summarize all rows.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Filter by seed (e.g. 42). If not set, summarize all rows.",
    )
    args = ap.parse_args()

    if not os.path.isfile(args.csv):
        print(f"Error: CSV not found: {args.csv}")
        print("Run the baseline first: bash scripts/run_baseline.sh")
        return 1

    rows = load_client_accuracy(args.csv)

    if args.alpha is not None:
        rows = [r for r in rows if float(r["alpha"]) == args.alpha]
    if args.seed is not None:
        rows = [r for r in rows if int(r["seed"]) == args.seed]

    if not rows:
        print("No rows match the given alpha/seed.")
        return 1

    # Group by (alpha, seed) for multi-run CSV
    groups = defaultdict(list)
    for r in rows:
        key = (str(r["alpha"]), int(r["seed"]))
        groups[key].append(r)

    os.makedirs(args.output_dir, exist_ok=True)

    # Build report: one summary per (alpha, seed), then overall
    report = {"runs": {}, "output_dir": args.output_dir, "csv_path": args.csv}

    for (alpha, seed), run_rows in sorted(groups.items()):
        s = summarize(run_rows)
        run_key = f"alpha{alpha}_seed{seed}"
        report["runs"][run_key] = s

        # Per-run text file
        txt_path = os.path.join(args.output_dir, f"utility_summary_{run_key}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"# Utility summary (FAZ 1)\n")
            f.write(f"# alpha={alpha}, seed={seed}\n")
            f.write(f"# Source: {args.csv}\n\n")
            f.write(f"Number of clients: {s['num_clients']}\n")
            f.write(f"Mean accuracy:     {s['mean_accuracy']:.4f}\n")
            f.write(f"Std accuracy:      {s['std_accuracy']:.4f}\n")
            f.write(f"Min accuracy:      {s['min_accuracy']:.4f}\n")
            f.write(f"Max accuracy:      {s['max_accuracy']:.4f}\n\n")
            f.write("Per-client accuracy:\n")
            f.write("-" * 50 + "\n")
            for p in s["per_client"]:
                f.write(f"  Client {p['client_id']:2d}: acc={p['local_test_accuracy']:.4f}  n_train={p['num_train_samples']:5d}  n_proto={p['num_proto_classes']}\n")
            f.write("-" * 50 + "\n")
        print(f"Wrote {txt_path}")

    # Overall (all rows in selection) summary
    overall = summarize(rows)
    report["overall"] = overall

    # Combined "latest" summary (one file for quick view)
    latest_path = os.path.join(args.output_dir, "utility_summary_latest.txt")
    if len(groups) == 1:
        run_key = list(report["runs"].keys())[0]
        s = report["runs"][run_key]
        with open(latest_path, "w", encoding="utf-8") as f:
            f.write(f"# Utility summary (FAZ 1) — latest run\n")
            f.write(f"# {run_key}\n")
            f.write(f"# Source: {args.csv}\n\n")
            f.write(f"Mean accuracy: {s['mean_accuracy']:.4f}\n")
            f.write(f"Std:           {s['std_accuracy']:.4f}\n")
            f.write(f"Min / Max:     {s['min_accuracy']:.4f} / {s['max_accuracy']:.4f}\n")
            f.write(f"Clients:       {s['num_clients']}\n")
    else:
        with open(latest_path, "w", encoding="utf-8") as f:
            f.write("# Utility summary (FAZ 1) — all runs in CSV\n")
            f.write(f"# Source: {args.csv}\n\n")
            for run_key, s in report["runs"].items():
                f.write(f"--- {run_key} ---\n")
                f.write(f"  Mean accuracy: {s['mean_accuracy']:.4f}  (std={s['std_accuracy']:.4f}, n={s['num_clients']})\n")
            f.write("\n")
    print(f"Wrote {latest_path}")

    # JSON (full report)
    json_path = os.path.join(args.output_dir, "utility_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {json_path}")

    # Console
    print("\n--- Utility summary ---")
    for run_key, s in report["runs"].items():
        print(f"  {run_key}: mean_acc={s['mean_accuracy']:.4f}  std={s['std_accuracy']:.4f}  clients={s['num_clients']}")
    print(f"\nOutputs saved under: {args.output_dir}")
    return 0


if __name__ == "__main__":
    exit(main())
