import torch
import numpy as np
import random
from typing import Dict, Any, List
from src.security.defenses.base import BaseDefense

class DummyPrototypeDefense(BaseDefense):
    """
    Dummy Prototype Defense:
    For each client, real class prototypes are identified and passed unchanged.
    For missing classes, a dummy prototype is created by:
    1. Linearly interpolating between two randomly selected real prototypes
       (coefficient between 0.5 and 0.9).
       (If only one real class, second target is a random unit vector).
    2. Adding Gaussian jitter (tau=0.02).
    3. Normalizing to unit norm.
    Outputs exactly 10 prototypes and 10 counts per client.
    """
    def __init__(self, num_classes: int = 10, tau: float = 0.02):
        self.num_classes = num_classes
        self.tau = tau
        self.last_stats = {}

    def apply(self, prototypes: Dict[int, torch.Tensor], counts: Dict[int, int] = None, **kwargs) -> Dict[int, np.ndarray]:
        defended_protos = {}
        real_classes = list(prototypes.keys())
        all_classes = list(range(self.num_classes))
        
        real_p_tensors = [
            p if torch.is_tensor(p) else torch.from_numpy(p)
            for p in prototypes.values()
        ]
        
        # Feature dimension (for random unit vector generation if needed)
        feature_dim = real_p_tensors[0].shape[0] if len(real_p_tensors) > 0 else 512
        device = real_p_tensors[0].device if len(real_p_tensors) > 0 else torch.device("cpu")
        
        # 1. Pass real classes unchanged
        for c in real_classes:
            p_c = prototypes[c]
            defended_protos[c] = p_c.cpu().numpy() if torch.is_tensor(p_c) else p_c
            
        missing_classes = [c for c in all_classes if c not in real_classes]
        
        # 2. Fill missing classes with dummies
        for c in missing_classes:
            # Sample fake counts randomly between 0 and 50
            if counts is not None:
                counts[c] = random.randint(1, 50)
                
            if len(real_p_tensors) >= 2:
                p1, p2 = random.sample(real_p_tensors, 2)
            elif len(real_p_tensors) == 1:
                p1 = real_p_tensors[0]
                p2 = torch.randn_like(p1)
                p2 = p2 / (torch.norm(p2) + 1e-8)
            else:
                p1 = torch.randn(feature_dim, device=device)
                p1 = p1 / (torch.norm(p1) + 1e-8)
                p2 = torch.randn(feature_dim, device=device)
                p2 = p2 / (torch.norm(p2) + 1e-8)
                
            # Mixing coefficient [0.5, 0.9]
            lam = random.uniform(0.5, 0.9)
            
            # Interpolation
            dummy = lam * p1 + (1.0 - lam) * p2
            
            # Gaussian jitter
            noise = torch.randn_like(dummy) * self.tau
            dummy = dummy + noise
            
            # Normalize to unit norm
            dummy = dummy / (torch.norm(dummy) + 1e-8)
            
            defended_protos[c] = dummy.cpu().numpy()
            
        self.last_stats = {
            "num_real": len(real_classes),
            "num_dummies": len(missing_classes)
        }
        
        return defended_protos, counts
