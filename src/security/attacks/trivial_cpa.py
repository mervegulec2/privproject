import numpy as np
from typing import Dict, Any, List
from src.security.base import BaseAttack
from src.security.metrics import get_categorical_metrics
from src.data_utils import load_split

class TrivialClassPresenceAttack(BaseAttack):
    """
    Professional Class Presence Auditor.
    Baseline: Protocol Leakage via shared prototype keys.
    Verification: Compares guesses against ground-truth training splits.
    """
    def __init__(self, num_classes: int = 10):
        self.num_classes = num_classes

    def execute(self, model_state: Dict[str, Any], shared_data: Dict[str, Any]) -> Dict[str, Any]:
        clients = shared_data.get("clients", [])
        if not clients:
            return {"status": "skipped", "reason": "No client data"}

        # Load Ground Truth
        splits = load_split(shared_data.get("split_path", "outputs/data/client_splits.npy"))
        if splits is None:
            return {"status": "error", "reason": "No split ground truth found"}

        audit_results = {}
        
        for client_info in clients:
            cid = client_info["cid"]
            observed_protos = client_info["protos"] 
            
            # 1. Attacker's Guess (Key-based)
            sent_classes = set([int(k) for k in observed_protos.keys()])
            y_pred = [1 if c in sent_classes else 0 for c in range(self.num_classes)]
            
            # 2. Ground Truth (From data splits)
            # Normalize cid to int — Flower metrics stores cid as string,
            # but load_split returns int keys
            try:
                cid_int = int(cid)
            except (ValueError, TypeError):
                continue
            
            if cid_int not in splits:
                continue
            
            member_indices = splits[cid_int]
                
            from src.data_utils import load_cifar10, Cifar10Config
            train_ds, _ = load_cifar10(Cifar10Config(root="data"))
            
            true_classes = set([train_ds.targets[i] for i in member_indices])
            y_true = [1 if c in true_classes else 0 for c in range(self.num_classes)]
            
            # 3. Calculate Scientific Metrics
            metrics = get_categorical_metrics(y_true, y_pred)
            
            audit_results[f"client_{cid}"] = {
                "f1_score": metrics["f1_score"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "n_true_classes": len(true_classes),
                "n_detected_classes": len(sent_classes),
                "true_positive_classes": sorted(list(true_classes.intersection(sent_classes))),
                "false_positive_classes": sorted(list(sent_classes - true_classes))
            }

        return audit_results
