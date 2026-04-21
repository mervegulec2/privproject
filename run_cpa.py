#!/usr/bin/env python3
"""
CPA attack runner script.
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run CPA attack on FL runs")
    parser.add_argument("run_dir", help="Path to the run directory")
    parser.add_argument("--type", choices=["trivial", "learned"], default="trivial",
                       help="CPA attack type")
    parser.add_argument("--save", action="store_true",
                       help="Save results")

    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Error: Run directory {run_dir} does not exist")
        sys.exit(1)

    from src.attacks.cpa.cpa_eval import evaluate_cpa, print_cpa_report, run_learned_cpa
    from src.attacks.common_io import save_attack_results

    if args.type == "learned":
        result = run_learned_cpa(str(run_dir))
    else:
        result = evaluate_cpa(str(run_dir), args.type)

    print_cpa_report({"attack_type": args.type, "num_runs": 1,
                     "aggregated_metrics": {k: {"mean": v, "std": 0} for k, v in result.metrics.items()},
                     "individual_results": [result], "dataset_stats": []})

    if args.save:
        save_attack_results(str(run_dir), result, f"cpa_{args.type}")
        print(f"\nResults saved to {run_dir}/attacks/")


if __name__ == "__main__":
    main()