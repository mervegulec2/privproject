import os
import csv
from datetime import datetime
from typing import Dict, List, Any

def setup_metrics_dir(path: str = "outputs/metrics"):
    if not os.path.exists(path):
        os.makedirs(path)

def save_round_to_csv(round_idx: int, metrics: Dict[str, Any], file_path: str = "outputs/metrics/simulation_results.csv"):
    """
    Saves aggregated and per-client metrics for a single round to a CSV file.
    metrics should contain:
        - avg_global, avg_local_prop, avg_local_aware, overall_avg
        - client_0_acc, client_1_acc, ... (and other per-client metrics)
    """
    setup_metrics_dir(os.path.dirname(file_path))
    
    file_exists = os.path.isfile(file_path)
    
    # Identify all keys to define the header
    header = ["timestamp", "round", "avg_global", "avg_local_prop", "avg_local_aware", "overall_avg"]
    
    # Pre-define client columns for stability (0-9)
    client_cols = []
    for i in range(10):
        client_cols.extend([f"client_{i}_global", f"client_{i}_local_prop", f"client_{i}_local_aware"])
    
    full_header = header + client_cols
    
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "round": round_idx,
    }
    # Only add metrics that are in our header
    for k in full_header:
        if k in metrics:
            row[k] = metrics[k]
    
    with open(file_path, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=full_header, extrasaction='ignore')
        if not file_exists or os.stat(file_path).st_size == 0:
            writer.writeheader()
        writer.writerow(row)

def save_summary_json(history_data: Dict[str, Any], file_path: str = "outputs/metrics/utility_summary.json"):
    """Saves final simulation summary to a JSON file."""
    import json
    setup_metrics_dir(os.path.dirname(file_path))
    with open(file_path, 'w') as f:
        json.dump(history_data, f, indent=4)
