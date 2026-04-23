import torch
import numpy as np
from typing import Dict, Optional
from .dp_gaussian import GaussianDPDefense

class AdaptiveGaussianDPDefense(GaussianDPDefense):
    """
    Implements Adaptive Differential Privacy for shared prototypes.
    
    Formula:
    sigma_c = min(sigma_max, alpha / sqrt(n_c + beta))
    
    Logic:
    - Classes with FEWER samples (small n_c) get MORE noise.
    - Classes with MORE samples (large n_c) get LESS noise.
    """
    def __init__(self, 
                 clip_norm: float = 1.0, 
                 alpha: float = 0.5, 
                 beta: float = 1.0, 
                 sigma_max: float = 1.0,
                 num_classes: int = 10):
        # Initialize parent with dummy sigma; we will calculate adaptive sigma per class
        super().__init__(clip_norm=clip_norm, sigma=0.0, num_classes=num_classes)
        self.alpha = alpha
        self.beta = beta
        self.sigma_max = sigma_max

    def apply(self, prototypes: Dict[int, torch.Tensor], counts: Dict[int, int] = None) -> Dict[int, torch.Tensor]:
        """
        Applies clipping and ADAPTIVE noise to each class prototype.
        """
        if counts is None:
            # Fallback to fixed sigma if counts are missing (safety)
            print("[Warning] Adaptive DP called without counts. Using alpha as fixed sigma.")
            self.sigma = self.alpha
            return super().apply(prototypes)

        defended_protos = {}
        
        for cls_id, proto in prototypes.items():
            # 1. Get sample count for this class
            n_c = counts.get(cls_id, 0)
            
            # 2. Compute Adaptive Sigma
            sigma_c = min(self.sigma_max, self.alpha / np.sqrt(n_c + self.beta))
            
            # 3. Clip & Add Noise
            if not isinstance(proto, torch.Tensor):
                p_tensor = torch.tensor(proto, dtype=torch.float32)
            else:
                p_tensor = proto.clone().detach()

            # L2 Clipping (Fixed threshold C)
            norm = torch.norm(p_tensor, p=2)
            if self.clip_norm is not None and norm > self.clip_norm:
                p_tensor = p_tensor * (self.clip_norm / (norm + 1e-8))

            # Add Gaussian Noise N(0, sigma_c^2 I)
            noise = torch.randn_like(p_tensor) * sigma_c
            p_tensor = p_tensor + noise

            defended_protos[cls_id] = p_tensor.cpu().numpy()

        return defended_protos

    def __repr__(self):
        return f"AdaptiveGaussianDPDefense(clip_norm={self.clip_norm}, alpha={self.alpha}, beta={self.beta})"
