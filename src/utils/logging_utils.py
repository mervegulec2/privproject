import os
import csv
from datetime import datetime
from typing import Dict, List, Any

def setup_metrics_dir(path: str = "outputs/metrics"):
    if not os.path.exists(path):
        os.makedirs(path)

def clear_metrics(file_path: str = "outputs/metrics/simulation_results.csv"):
    """Deletes existing metrics file to start fresh."""
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Cleared old metrics at: {file_path}")

def save_round_to_csv(round_idx: int, test_type: str, avg_acc: float, client_metrics: Dict[int, float], file_path: str = "outputs/metrics/simulation_results.csv"):
    """
    Saves metrics for a specific test type in a round to a CSV file.
    Creates 1 row per round per test_type.
    """
    setup_metrics_dir(os.path.dirname(file_path))
    
    file_exists = os.path.isfile(file_path)
    
    # Header: timestamp, round, test_type, avg_accuracy, client_0, client_1, ...
    header = ["timestamp", "round", "test_type", "avg_accuracy"]
    client_cols = [f"client_{i}" for i in range(10)]
    full_header = header + client_cols
    
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "round": round_idx,
        "test_type": test_type,
        "avg_accuracy": avg_acc
    }
    
    # Map client metrics to columns
    for cid, acc in client_metrics.items():
        col_name = f"client_{cid}"
        if col_name in full_header:
            row[col_name] = acc
    
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
