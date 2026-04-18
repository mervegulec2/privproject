from __future__ import annotations
import os
import numpy as np
from typing import Dict
from dataclasses import dataclass
from torchvision import datasets, transforms

@dataclass(frozen=True)
class Cifar10Config:
    root: str = "data"
    num_classes: int = 10

def get_cifar10_transforms():
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, test_tf

def load_cifar10(cfg: Cifar10Config):
    train_tf, test_tf = get_cifar10_transforms()
    train_ds = datasets.CIFAR10(root=cfg.root, train=True, download=True, transform=train_tf)
    test_ds  = datasets.CIFAR10(root=cfg.root, train=False, download=True, transform=test_tf)
    return train_ds, test_ds

@dataclass(frozen=True)
class DirichletSplitConfig:
    num_clients: int = 15
    alpha: float = 0.1
    seed: int = 42
    num_classes: int = 10
    min_size_per_client: int = 500

def _targets(dataset) -> np.ndarray:
    if hasattr(dataset, "targets"):
        return np.array(dataset.targets, dtype=np.int64)
    raise ValueError("Dataset missing 'targets'")

def dirichlet_split_indices(dataset, cfg: DirichletSplitConfig) -> Dict[int, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    y = _targets(dataset)
    class_indices = [np.where(y == c)[0] for c in range(cfg.num_classes)]

    for _ in range(200):
        buckets = {i: [] for i in range(cfg.num_clients)}
        for c in range(cfg.num_classes):
            idx = class_indices[c].copy()
            rng.shuffle(idx)
            proportions = rng.dirichlet([cfg.alpha] * cfg.num_clients)
            counts = (proportions * len(idx)).astype(int)
            diff = len(idx) - counts.sum()
            if diff != 0:
                for k in rng.choice(cfg.num_clients, size=abs(diff), replace=True):
                    counts[k] += 1 if diff > 0 else -1
            start = 0
            for cid, cnt in enumerate(counts):
                if cnt > 0:
                    buckets[cid].append(idx[start:start+cnt])
                    start += cnt

        client_map = {}
        sizes = []
        for cid in range(cfg.num_clients):
            arr = np.concatenate(buckets[cid]).astype(np.int64) if buckets[cid] else np.array([], dtype=np.int64)
            client_map[cid] = arr
            sizes.append(len(arr))

        if min(sizes) >= cfg.min_size_per_client and sum(sizes) == len(dataset):
            return client_map
    raise RuntimeError("Could not satisfy min_size_per_client.")

def save_split(client_map: Dict[int, np.ndarray], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr = np.empty((len(client_map),), dtype=object)
    for cid in sorted(client_map.keys()):
        arr[cid] = client_map[cid]
    np.save(path, arr, allow_pickle=True)

def load_split(path: str) -> Dict[int, np.ndarray]:
    arr = np.load(path, allow_pickle=True)
    return {cid: arr[cid] for cid in range(len(arr))}

def get_seen_classes(dataset, indices: np.ndarray) -> set[int]:
    """Returns the set of unique class labels present in a subset of the dataset."""
    y = _targets(dataset)
    subset_y = y[indices]
    return set(np.unique(subset_y).tolist())
