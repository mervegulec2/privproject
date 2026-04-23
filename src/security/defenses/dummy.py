import torch
import numpy as np
import random
from typing import Dict, Any, List
from src.security.defenses.base import BaseDefense

class DummyPrototypeDefense(BaseDefense):
    """
    Dummy Prototype Defense (Final Version):
    For each real class prototype, generates an additional cross-class decoy.
    
    Formula: d_c = lambda * p_c + (1 - lambda) * v_j + Normal(0, tau^2 I)
    """
    def __init__(self, num_classes: int = 10, lambda_val: float = 0.7, tau: float = 0.02, dummy_ratio: int = 1):
        self.num_classes = num_classes
        self.lambda_val = lambda_val
        self.tau = tau
        self.dummy_ratio = dummy_ratio
        self.last_stats = {}

    def apply(self, prototypes: Dict[int, torch.Tensor], **kwargs) -> Dict[int, List[np.ndarray]]:
        defended_protos = {}
        local_classes = list(prototypes.keys())
        all_class_ids = list(range(self.num_classes))
        
        local_p_tensors = [p if torch.is_tensor(p) else torch.from_numpy(p) for p in prototypes.values()]
        mean_local_p = torch.stack(local_p_tensors).mean(dim=0)
        
        stats_data = {"similarities": [], "distortions": []}
        
        for cls_id, p_real in prototypes.items():
            p_c = p_real if torch.is_tensor(p_real) else torch.from_numpy(p_real)
            candidates = [p_c.cpu().numpy()]
            
            for _ in range(self.dummy_ratio):
                j = random.choice([cid for cid in all_class_ids if cid != cls_id])
                
                if j in prototypes:
                    v_j = prototypes[j]
                    if not torch.is_tensor(v_j): v_j = torch.from_numpy(v_j)
                else:
                    v_j = mean_local_p
                
                # Formula
                dummy = self.lambda_val * p_c + (1.0 - self.lambda_val) * v_j
                noise = torch.randn_like(dummy) * self.tau
                dummy = dummy + noise
                
                # Metrics Calculation
                cos_sim = torch.nn.functional.cosine_similarity(p_c.unsqueeze(0), dummy.unsqueeze(0)).item()
                l2_dist = torch.norm(p_c - dummy).item()
                stats_data["similarities"].append(cos_sim)
                stats_data["distortions"].append(l2_dist)
                
                candidates.append(dummy.cpu().numpy())
            
            random.shuffle(candidates)
            defended_protos[cls_id] = candidates
            
        if stats_data["similarities"]:
            self.last_stats = {
                "avg_dummy_similarity": float(np.mean(stats_data["similarities"])),
                "avg_dummy_distortion": float(np.mean(stats_data["distortions"]))
            }
        
        return defended_protos
