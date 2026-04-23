import torch
import numpy as np
import random
from typing import Dict, Any, Optional
from src.security.defenses.base import BaseDefense

class DummyPrototypeDefense(BaseDefense):
    """
    Dummy Prototype Defense:
    For each real class prototype, generates an additional cross-class decoy.
    The decoy is an interpolation between the real class and another class.
    
    Formula: d_c = lambda * p_c + (1 - lambda) * p_j + Normal(0, tau^2 I)
    """
    def __init__(self, lambda_val: float = 0.7, tau: float = 0.02, dummy_ratio: int = 1):
        self.lambda_val = lambda_val
        self.tau = tau
        self.dummy_ratio = dummy_ratio

    def apply(self, prototypes: Dict[int, torch.Tensor], **kwargs) -> Dict[int, Any]:
        """
        Generates dummy prototypes for each existing class.
        Returns a dictionary where values are Lists [real, dummy1, ...].
        """
        defended_protos = {}
        available_classes = list(prototypes.keys())
        
        for cls_id, p_real in prototypes.items():
            # Ensure p_real is a tensor for math
            p_tensor = p_real if torch.is_tensor(p_real) else torch.from_numpy(p_real)
            
            # Start with the real one
            candidates = [p_tensor.cpu().numpy()]
            
            # Generate dummies
            if len(available_classes) > 1:
                for _ in range(self.dummy_ratio):
                    # Pick a different class j != c from available local classes
                    j = random.choice([c for c in available_classes if c != cls_id])
                    p_j = prototypes[j]
                    p_j_tensor = p_j if torch.is_tensor(p_j) else torch.from_numpy(p_j)
                    
                    # Interpolation
                    dummy = self.lambda_val * p_tensor + (1 - self.lambda_val) * p_j_tensor
                    
                    # Add jitter
                    noise = torch.randn_like(dummy) * self.tau
                    dummy = dummy + noise
                    
                    candidates.append(dummy.cpu().numpy())
            
            # Store as list (to be aggregated by PrototypeStrategy)
            defended_protos[cls_id] = candidates
            
        return defended_protos
