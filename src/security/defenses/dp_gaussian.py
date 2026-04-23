import torch
import numpy as np
from typing import Dict
from .base import BaseDefense

class GaussianDPDefense(BaseDefense):
    """
    Implements Differential Privacy for shared prototypes.
    Process:
    1. L2 Clipping: ensures the sensitivity of each prototype is bounded by 'clip_norm'.
    2. Gaussian Noise: adds noise scaled by 'sigma' to satisfy (epsilon, delta)-DP.
    """
    def __init__(self, clip_norm: float = 1.0, sigma: float = None, epsilon: float = None, delta: float = 1e-5, num_classes: int = 10):
        self.clip_norm = clip_norm
        self.delta = delta
        self.num_classes = num_classes
        
        if epsilon is not None:
            # Calculate sigma from epsilon for a single-round Gaussian Mechanism
            # Formula: sigma = (clip_norm * sqrt(2 * ln(1.25 / delta))) / epsilon
            self.epsilon = epsilon
            self.sigma = (clip_norm * np.sqrt(2 * np.log(1.25 / delta))) / epsilon
        else:
            self.sigma = sigma if sigma is not None else 0.01
            self.epsilon = None # Could calculate backward if needed

    def apply(self, prototypes: Dict[int, torch.Tensor], **kwargs) -> Dict[int, torch.Tensor]:
        """
        Applies clipping and noise to each class prototype.
        """
        if self.sigma <= 0 and self.clip_norm is None:
            return prototypes

        defended_protos = {}
        
        for cls_id, proto in prototypes.items():
            # Ensure we are working with a torch tensor on the correct device
            if not isinstance(proto, torch.Tensor):
                p_tensor = torch.tensor(proto, dtype=torch.float32)
            else:
                p_tensor = proto.clone().detach()

            # 1. L2 Clipping: P' = P * min(1, C / ||P||_2)
            norm = torch.norm(p_tensor, p=2)
            if norm > self.clip_norm:
                p_tensor = p_tensor * (self.clip_norm / (norm + 1e-8))

            # 2. Gaussian Noise: P'' = P' + N(0, sigma^2 * I)
            # Standard DP sensitivity for the mean of clipped vectors is (clip_norm / n)
            # but since we are sharing prototypes directly, we treat the sensitivity as clip_norm.
            noise = torch.randn_like(p_tensor) * self.sigma
            p_tensor = p_tensor + noise

            # Convert back to numpy for pipeline compatibility if needed
            defended_protos[cls_id] = p_tensor.cpu().numpy()

        return defended_protos

    def __repr__(self):
        return f"GaussianDPDefense(clip_norm={self.clip_norm}, sigma={self.sigma})"
