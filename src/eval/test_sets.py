import numpy as np
from torch.utils.data import Dataset, Subset
from typing import List, Dict, Set

def get_indices_by_class(dataset: Dataset, num_classes: int = 10) -> Dict[int, np.ndarray]:
    """Helper to group dataset indices by their class label."""
    if hasattr(dataset, "targets"):
        targets = np.array(dataset.targets)
    else:
        # Fallback for Subsets or other datasets
        targets = np.array([dataset[i][1] for i in range(len(dataset))])
    
    indices_by_class = {}
    for c in range(num_classes):
        indices_by_class[c] = np.where(targets == c)[0]
    return indices_by_class

def create_balanced_test_indices(
    dataset: Dataset, 
    samples_per_class: int = 100, 
    seed: int = 42
) -> np.ndarray:
    """Creates a balanced global test set."""
    rng = np.random.default_rng(seed)
    class_indices = get_indices_by_class(dataset)
    
    selected_indices = []
    for c, idxs in class_indices.items():
        if len(idxs) == 0:
            continue
        n = min(len(idxs), samples_per_class)
        selected = rng.choice(idxs, size=n, replace=False)
        selected_indices.extend(selected)
    
    return np.array(selected_indices)

def create_local_aware_indices(
    dataset: Dataset, 
    seen_classes: Set[int]
) -> np.ndarray:
    """Creates a test set containing only classes present in training."""
    class_indices = get_indices_by_class(dataset)
    
    selected_indices = []
    for c in seen_classes:
        if c in class_indices:
            selected_indices.extend(class_indices[c])
    
    return np.array(selected_indices)

def create_local_aware_balanced_indices(
    dataset: Dataset, 
    seen_classes: Set[int],
    samples_per_class: int = 100,
    seed: int = 42
) -> np.ndarray:
    """Creates a balanced test set containing only classes present in training."""
    rng = np.random.default_rng(seed)
    class_indices = get_indices_by_class(dataset)
    
    selected_indices = []
    for c in seen_classes:
        if c in class_indices:
            idxs = class_indices[c]
            n = min(len(idxs), samples_per_class)
            selected = rng.choice(idxs, size=n, replace=False)
            selected_indices.extend(selected)
    
    return np.array(selected_indices)
