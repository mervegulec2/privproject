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
    
    v_j (Source Vector) rules:
    - Case A: If class j exists locally, v_j = p_j
    - Case B: If class j does NOT exist locally, v_j = mean(all local prototypes)
    """
    def __init__(self, num_classes: int = 10, lambda_val: float = 0.7, tau: float = 0.02, dummy_ratio: int = 1):
        self.num_classes = num_classes
        self.lambda_val = lambda_val
        self.tau = tau
        self.dummy_ratio = dummy_ratio

    def apply(self, prototypes: Dict[int, torch.Tensor], **kwargs) -> Dict[int, List[np.ndarray]]:
        """
        Generates dummy prototypes following the v_j source vector rules.
        """
        defended_protos = {}
        local_classes = list(prototypes.keys())
        all_class_ids = list(range(self.num_classes))
        
        # Pre-compute the mean of all local prototypes for Case B
        local_p_tensors = [p if torch.is_tensor(p) else torch.from_numpy(p) for p in prototypes.values()]
        mean_local_p = torch.stack(local_p_tensors).mean(dim=0)
        
        for cls_id, p_real in prototypes.items():
            p_c = p_real if torch.is_tensor(p_real) else torch.from_numpy(p_real)
            
            # Start with the real one (converted to numpy for transmission)
            candidates = [p_c.cpu().numpy()]
            
            for _ in range(self.dummy_ratio):
                # 1. Randomly choose another class j != c from the full global set {0..9}
                j = random.choice([cid for cid in all_class_ids if cid != cls_id])
                
                # 2. Determine v_j
                if j in prototypes:
                    # Case A: Class j exists locally
                    v_j = prototypes[j]
                    if not torch.is_tensor(v_j):
                        v_j = torch.from_numpy(v_j)
                else:
                    # Case B: Class j does NOT exist locally
                    v_j = mean_local_p
                
                # 3. Apply mathematical formula: d_c = lambda*p_c + (1-lambda)*v_j + noise
                dummy = self.lambda_val * p_c + (1.0 - self.lambda_val) * v_j
                
                # Add small Gaussian noise: Normal(0, tau^2 I)
                noise = torch.randn_like(dummy) * self.tau
                dummy = dummy + noise
                
                candidates.append(dummy.cpu().numpy())
            
            # Shuffle candidates so the server doesn't know which one is real (index 0 is not always real)
            random.shuffle(candidates)
            
            defended_protos[cls_id] = candidates
            
        return defended_protos
