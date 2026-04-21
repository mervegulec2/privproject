"""MIA evaluator (strict server-visible protocol).

This evaluator is feasibility-first. Under the current strict protocol,
record-level MIA is generally unsupported because the server lacks any
sample-level representation.
"""

from typing import Dict, Any, List

from .mia_dataset import analyze_run_feasibility


def evaluate_mia(run_dir: str) -> Dict[str, Any]:
    report = analyze_run_feasibility(run_dir)
    return {
        "attack_mode": "feasibility",
        "status": report.status,  # "unsupported"/"limited"/"supported" style is surfaced via status+reason
        "reason": report.reason,
        "required_future_artifacts": report.required_future_artifacts,
        "available_signals": report.available_signals,
        "recommended_approach": report.recommended_approach,
        "feasibility_report": report,
    }


def evaluate_mia_multiple_runs(run_dirs: List[str]) -> Dict[str, Any]:
    reports = [evaluate_mia(d) for d in run_dirs]
    statuses = [r["status"] for r in reports]
    reasons = sorted(set(r["reason"] for r in reports))
    summary_status = "mixed" if len(set(statuses)) > 1 else statuses[0]
    return {
        "attack_mode": "feasibility",
        "num_runs": len(run_dirs),
        "summary_status": summary_status,
        "common_reasons": reasons,
        "individual_reports": reports,
    }