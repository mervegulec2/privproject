"""Master runner to execute all attack modules (CPA, MIA, etc.).

Currently dispatches to `run_cpa.py` and `run_mia.py`.
"""

import subprocess
import sys


def main(args):
    # Simple orchestrator: call run_cpa then run_mia
    run_dirs = args[1:]
    if not run_dirs:
        print("Usage: python run_attacks.py <run_dir> [<run_dir> ...]")
        return

    # Run CPA
    print("Running CPA on:", run_dirs)
    subprocess.check_call([sys.executable, "run_cpa.py"] + run_dirs)

    # Run MIA
    print("Running MIA feasibility on:", run_dirs)
    subprocess.check_call([sys.executable, "run_mia.py"] + run_dirs)


if __name__ == "__main__":
    main(sys.argv)
#!/usr/bin/env python3
"""
Main attack runner script.
Orchestrates CPA and MIA attacks on logged FL runs.
"""

import argparse
import sys
from pathlib import Path
from typing import List


def main():
    parser = argparse.ArgumentParser(description="Run privacy attacks on FL runs")
    parser.add_argument("run_dir", help="Path to the run directory")
    parser.add_argument("--attack", choices=["cpa", "mia", "all"], default="all",
                       help="Attack type to run")
    parser.add_argument("--cpa-type", choices=["trivial"], default="trivial",
                       help="CPA attack type")
    parser.add_argument("--mia-mode", choices=["feasibility"], default="feasibility",
                       help="MIA attack mode")
    parser.add_argument("--save-results", action="store_true",
                       help="Save results to run directory")

    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Error: Run directory {run_dir} does not exist")
        sys.exit(1)

    # Import attack modules
    from src.attacks.cpa.cpa_eval import evaluate_cpa, print_cpa_report
    from src.attacks.mia.mia_eval import evaluate_mia, print_mia_report
    from src.attacks.common_io import save_attack_results

    results = {}

    if args.attack in ["cpa", "all"]:
        print("Running CPA attack...")
        cpa_result = evaluate_cpa(str(run_dir), args.cpa_type)
        results["cpa"] = cpa_result

        print_cpa_report({"attack_type": args.cpa_type, "num_runs": 1,
                         "aggregated_metrics": {k: {"mean": v, "std": 0} for k, v in cpa_result.metrics.items()},
                         "individual_results": [cpa_result], "dataset_stats": []})

    if args.attack in ["mia", "all"]:
        print("Running MIA feasibility assessment...")
        mia_result = evaluate_mia(str(run_dir), args.mia_mode)
        results["mia"] = mia_result

        print_mia_report(mia_result)

    if args.save_results:
        for attack_name, result in results.items():
            save_attack_results(str(run_dir), result, attack_name)
        print(f"\nResults saved to {run_dir}/attacks/")


if __name__ == "__main__":
    main()