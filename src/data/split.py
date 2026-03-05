from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import os
import numpy as np

@dataclass(frozen=True)
class DirichletSplitConfig:
    num_clients: int = 15
    alpha: float = 0.1  # smaller = stronger non-IID (fewer classes per client); 0.1 for class-presence attacks
    seed: int = 42
    num_classes: int = 10
    min_size_per_client: int = 500  # ~3k hedef için güvenli; gerekirse düşür

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
                # fix rounding
                for k in rng.choice(cfg.num_clients, size=abs(diff), replace=True):
                    counts[k] += 1 if diff > 0 else -1

            start = 0
            for cid, cnt in enumerate(counts):
                if cnt <= 0:
                    continue
                buckets[cid].append(idx[start:start+cnt])
                start += cnt

        client_map = {}
        sizes = []
        for cid in range(cfg.num_clients):
            if len(buckets[cid]) == 0:
                arr = np.array([], dtype=np.int64)
            else:
                arr = np.concatenate(buckets[cid]).astype(np.int64)
            client_map[cid] = arr
            sizes.append(len(arr))

        if min(sizes) >= cfg.min_size_per_client and sum(sizes) == len(dataset):
            return client_map

    raise RuntimeError("Could not satisfy min_size_per_client. Lower it or change seed/alpha.")

def class_hist(dataset, indices: np.ndarray, num_classes: int = 10) -> np.ndarray:
    y = _targets(dataset)[indices]
    return np.bincount(y, minlength=num_classes)

def save_split(client_map: Dict[int, np.ndarray], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # save as object array: each element is np.ndarray indices
    arr = np.empty((len(client_map),), dtype=object)
    for cid in sorted(client_map.keys()):
        arr[cid] = client_map[cid]
    np.save(path, arr, allow_pickle=True)

def load_split(path: str) -> Dict[int, np.ndarray]:
    arr = np.load(path, allow_pickle=True)
    return {cid: arr[cid] for cid in range(len(arr))}