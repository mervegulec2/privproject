from __future__ import annotations
from typing import Dict
import numpy as np
import torch
from torch.utils.data import DataLoader

@torch.no_grad()
def compute_class_prototypes(model, loader: DataLoader, device: str, num_classes: int = 10) -> Dict[int, np.ndarray]:
    """
    Returns dict: {class_id: prototype_vector(512,)}
    Important: if a class is absent, it is NOT included in the dict.
    """
    model.to(device)
    model.eval()

    sums = {c: None for c in range(num_classes)}
    counts = {c: 0 for c in range(num_classes)}

    for x, y in loader:
        x = x.to(device)
        y = y.numpy()
        logits, emb = model(x)  # emb: [B,512]
        emb_np = emb.detach().cpu().numpy()

        for i in range(len(y)):
            c = int(y[i])
            if sums[c] is None:
                sums[c] = emb_np[i].copy()
            else:
                sums[c] += emb_np[i]
            counts[c] += 1

    prototypes = {}
    for c in range(num_classes):
        if counts[c] > 0:
            prototypes[c] = (sums[c] / counts[c]).astype(np.float32)

    return prototypes