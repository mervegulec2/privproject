from abc import ABC, abstractmethod
from typing import Dict, Any, List
import torch.nn as nn
import numpy as np

class BaseDefense(ABC):
    """Base class for all defense mechanisms (applied at client-side)."""
    @abstractmethod
    def apply(self, prototypes: Dict[int, np.ndarray], counts: Dict[int, int]) -> Dict[int, np.ndarray]:
        """Modifies prototypes before they are sent to the server."""
        pass

class BaseAttack(ABC):
    """Base class for all adversary attacks (executed by the curious server)."""
    @abstractmethod
    def execute(self, model_state: Dict[str, Any], shared_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs the attack using observed data.
        :param model_state: Current weights/architecture of the backbone.
        :param shared_data: Prototypes and counts observed from clients.
        """
        pass
