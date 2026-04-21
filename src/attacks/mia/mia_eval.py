"""MIA evaluator: feasibility report + placeholder metrics.

Produces a FeasibilityReport and, if applicable, dummy evaluation outputs.
"""

from typing import Dict, Any
from .mia_dataset import analyze_run_feasibility


def evaluate_mia(run_dir: str) -> Dict[str, Any]:
    report = analyze_run_feasibility(run_dir)
    out = {"feasibility": report}
    # If feasibility is limited but possible, return placeholder metrics
    if report.status == "limited":
        out["metrics"] = {"roc_auc": None, "pr_auc": None}
    return out
"""
MIA evaluation utilities.
"""

from typing import Dict, Any, List
from ..common_types import AttackResult, FeasibilityReport
from .mia_dataset import assess_mia_feasibility


def evaluate_mia_feasibility(run_dir: str) -> FeasibilityReport:
    """
    Evaluate MIA feasibility for a run.
    """
    return assess_mia_feasibility(run_dir)


def evaluate_mia(run_dir: str, attack_mode: str = "feasibility") -> Dict[str, Any]:
    """
    Evaluate MIA attack on a run directory.
    """
    if attack_mode == "feasibility":
        report = evaluate_mia_feasibility(run_dir)
        return {
            "attack_mode": attack_mode,
            "feasibility_report": report,
            "status": report.status,
            "reason": report.reason,
        }
    else:
        raise ValueError(f"Unknown MIA attack mode: {attack_mode}")


def evaluate_mia_multiple_runs(run_dirs: List[str], attack_mode: str = "feasibility") -> Dict[str, Any]:
    """
    Evaluate MIA across multiple runs.
    """
    reports = []
    for run_dir in run_dirs:
        result = evaluate_mia(run_dir, attack_mode)
        reports.append(result)

    # Summarize feasibility across runs
    statuses = [r["status"] for r in reports]
    reasons = list(set(r["reason"] for r in reports))

    summary_status = "mixed" if len(set(statuses)) > 1 else statuses[0]

    return {
        "attack_mode": attack_mode,
        "num_runs": len(run_dirs),
        "summary_status": summary_status,
        "common_reasons": reasons,
        "individual_reports": reports,
    }


def print_mia_report(result: Dict[str, Any]) -> None:
    """Print a formatted MIA evaluation report."""
    print("\n" + "="*60)
    print("MIA EVALUATION REPORT")
    print("="*60)
    print(f"Attack Mode: {result['attack_mode']}")
    print(f"Number of Runs: {result.get('num_runs', 1)}")
    print()

    if "feasibility_report" in result:
        report = result["feasibility_report"]
        print("FEASIBILITY ASSESSMENT:")
        print("-" * 40)
        print(f"Status: {report.status.upper()}")
        print(f"Reason: {report.reason}")
        print(f"Available Signals: {', '.join(report.available_signals)}")
        print(f"Recommended Approach: {report.recommended_approach}")
    elif "summary_status" in result:
        print("MULTI-RUN SUMMARY:")
        print("-" * 40)
        print(f"Overall Status: {result['summary_status'].upper()}")
        print("Common Reasons:")
        for reason in result["common_reasons"]:
            print(f"  - {reason}")
    print()