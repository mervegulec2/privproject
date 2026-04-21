"""Runner for membership inference feasibility/evaluation.

This script performs a feasibility check under strict server-visible artifacts,
and will only run placeholder evaluators if sufficient signals exist.
"""

import sys
from typing import List
from src.attacks.common_io import load_run_meta, list_rounds
from src.attacks.common_io import load_client_upload
from src.attacks.common_types import FeasibilityReport


def main(run_dirs: List[str]):
    for run_dir in run_dirs:
        print(f"Checking MIA feasibility for {run_dir}")
        meta = load_run_meta(run_dir)
        allowed = meta.get("sent_fields", {})

        # If only prototypes and counts are sent, sample-level MIA is limited
        if allowed.get("prototypes") and allowed.get("class_counts"):
            report = FeasibilityReport(
                attack_mode="strict_server_current_protocol",
                status="limited",
                reason="No sample-level representations are server-visible",
                available_signals=[k for k, v in allowed.items() if v],
                recommended_approach="feasibility_only"
            )
        else:
            report = FeasibilityReport(
                attack_mode="strict_server_current_protocol",
                status="failed",
                reason="Insufficient server-visible signals",
                available_signals=[k for k, v in allowed.items() if v],
                recommended_approach="none"
            )

        print("MIA Feasibility:")
        print(report)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_mia.py <run_dir> [<run_dir> ...]")
        sys.exit(1)
    main(sys.argv[1:])
#!/usr/bin/env python3
"""
MIA attack runner script.
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run MIA attack on FL runs")
    parser.add_argument("run_dir", help="Path to the run directory")
    parser.add_argument("--mode", choices=["feasibility"], default="feasibility",
                       help="MIA attack mode")
    parser.add_argument("--save", action="store_true",
                       help="Save results")

    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Error: Run directory {run_dir} does not exist")
        sys.exit(1)

    from src.attacks.mia.mia_eval import evaluate_mia, print_mia_report
    from src.attacks.common_io import save_attack_results

    result = evaluate_mia(str(run_dir), args.mode)

    print_mia_report(result)

    if args.save:
        save_attack_results(str(run_dir), result, f"mia_{args.mode}")
        print(f"\nResults saved to {run_dir}/attacks/")


if __name__ == "__main__":
    main()