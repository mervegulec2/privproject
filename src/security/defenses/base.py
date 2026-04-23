import abc
import torch
from typing import Dict

class BaseDefense(abc.ABC):
    """Abstract base class for all privacy-preserving defenses."""
    
    @abc.abstractmethod
    def apply(self, prototypes: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        """Apply the defense to a dictionary of prototypes."""
        pass

    def __call__(self, prototypes: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        return self.apply(prototypes)
