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

def create_local_proportional_indices(
    test_dataset: Dataset, 
    train_dataset: Dataset,
    client_train_indices: np.ndarray,
    total_test_samples: int = 1000,
    seed: int = 42
) -> np.ndarray:
    """Creates a test set where class proportions match the client's training data."""
    rng = np.random.default_rng(seed)
    
    if hasattr(train_dataset, "targets"):
        train_targets = np.array(train_dataset.targets)
    else:
        train_targets = np.array([train_dataset[i][1] for i in range(len(train_dataset))])
        
    client_labels = train_targets[client_train_indices]
    classes, counts = np.unique(client_labels, return_counts=True)
    proportions = counts / len(client_labels)
    
    test_class_indices = get_indices_by_class(test_dataset)
    
    selected_indices = []
    for c, prop in zip(classes, proportions):
        if c in test_class_indices:
            idxs = test_class_indices[c]
            n = int(np.round(prop * total_test_samples))
            n = min(len(idxs), n)
            if n > 0:
                selected = rng.choice(idxs, size=n, replace=False)
                selected_indices.extend(selected)
    
    return np.array(selected_indices)
