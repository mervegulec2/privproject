import torch
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.metrics import roc_auc_score
from src.security.base import BaseAttack
from src.security.metrics import get_inference_leakage, get_tpr_at_fpr
from src.models import ResNet18Cifar
from src.data_utils import load_cifar10, Cifar10Config, load_split

class ScientificMIAAuditor(BaseAttack):
    """
    A Full Membership Inference Auditor.
    Matches the 'Adversary Knowledge' Slide:
    - Observed: Prototypes
    - System: Global model init / Architecture
    - Auxiliary: CIFAR-10 Test set
    """
    def __init__(self, num_classes: int = 10, samples_per_group: int = 250):
        self.num_classes = num_classes
        self.samples_per_group = samples_per_group

    def _get_similarity(self, embeddings: torch.Tensor, prototype: torch.Tensor) -> torch.Tensor:
        """Calculates cosine similarity between embeddings and class prototype."""
        return torch.nn.functional.cosine_similarity(embeddings, prototype.unsqueeze(0))

    def execute(self, model_state: Dict[str, Any], shared_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculates AUC-ROC by comparing member scores vs non-member scores.
        """
        if "clients" not in shared_data:
            return {"status": "skipped", "reason": "No client data"}

        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 1. Setup Model (Strictly using System Knowledge: Initial Weights)
        model = ResNet18Cifar(num_classes=self.num_classes)
        if model_state is not None:
             model.load_state_dict(model_state)
        model.to(device)
        model.eval()

        # 2. Load Data
        train_ds, test_ds = load_cifar10(Cifar10Config(root="data"))
        
        audit_results = {}
        
        for client_info in shared_data["clients"]:
            cid = client_info["cid"]
            protos = client_info["protos"]
            
            # For a scientific audit, we need to know who the REAL members were.
            # We assume the evaluator has access to the split file.
            split = load_split("outputs/data/client_splits.json") # Helper to get indices
            if split is None or str(cid) not in split:
                continue
                
            member_indices = split[str(cid)]
            non_member_indices = list(range(len(test_ds))) # Non-members from global test pool
            
            # Sample groups
            m_idx = np.random.choice(member_indices, min(len(member_indices), self.samples_per_group), replace=False)
            nm_idx = np.random.choice(non_member_indices, min(len(non_member_indices), self.samples_per_group), replace=False)
            
            member_scores = []
            non_member_scores = []

            with torch.no_grad():
                for target_class, proto_np in protos.items():
                    proto = torch.tensor(proto_np, device=device).float()
                    
                    # Score Members of this class
                    curr_m_idx = [i for i in m_idx if train_ds.targets[i] == int(target_class)]
                    if curr_m_idx:
                        imgs = torch.stack([train_ds[i][0] for i in curr_m_idx]).to(device)
                        _, embs = model(imgs)
                        scores = self._get_similarity(embs, proto)
                        member_scores.extend(scores.cpu().tolist())
                        
                    # Score Non-Members of this class
                    curr_nm_idx = [i for i in nm_idx if test_ds.targets[i] == int(target_class)]
                    if curr_nm_idx:
                        imgs = torch.stack([test_ds[i][0] for i in curr_nm_idx]).to(device)
                        _, embs = model(imgs)
                        scores = self._get_similarity(embs, proto)
                        non_member_scores.extend(scores.cpu().tolist())

            # 3. Calculate Scientific Metrics
            if member_scores and non_member_scores:
                leakage = get_inference_leakage(member_scores, non_member_scores)
                tpr_val = get_tpr_at_fpr(member_scores, non_member_scores, fpr_target=0.01)
                
                audit_results[f"client_{cid}"] = {
                    "auc_roc": leakage["auc_roc"],
                    "attacker_advantage": leakage["attacker_advantage"],
                    "confidence_gap": leakage["score_gap"],
                    "tpr_at_1percent_fpr": tpr_val,
                    "n_members": len(member_scores),
                    "n_non_members": len(non_member_scores),
                    # Keep raw scores for aggregate plotting at the end
                    "member_scores": member_scores,
                    "non_member_scores": non_member_scores
                }

        return audit_results
