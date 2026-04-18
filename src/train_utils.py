import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from torch.utils.data import DataLoader
from tqdm import tqdm

@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 5
    batch_size: int = 64
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    device: str = "cpu"
    train_backbone: bool = True
    train_head: bool = True


def train_local_proto(
    model,
    train_loader: DataLoader,
    global_protos: dict,
    cfg: TrainConfig,
    lambda_p: float = 0.05,
    progress: bool = False,
):
    """Collaborative training with Prototype Alignment Loss."""
    model.to(cfg.device)
    model.train()

    # Personalization knobs: allow training only parts of the model (e.g., head-only).
    if hasattr(model, "backbone"):
        for p in model.backbone.parameters():
            p.requires_grad = bool(cfg.train_backbone)
    if hasattr(model, "fc"):
        for p in model.fc.parameters():
            p.requires_grad = bool(cfg.train_head)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        return 0.0

    opt = torch.optim.SGD(trainable_params, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    
    use_amp = (cfg.device == "cuda")
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # Convert global prototypes to torch tensors on correct device
    g_protos_torch = {int(c): torch.tensor(v, dtype=torch.float32, device=cfg.device) for c, v in global_protos.items()}
    # Build a fast lookup table for batch-wise prototype alignment:
    # proto_table[y] gives the prototype for class y (if available).
    proto_table = None
    proto_avail = None
    if len(g_protos_torch) > 0:
        d = next(iter(g_protos_torch.values())).numel()
        k = max(g_protos_torch.keys()) + 1
        proto_table = torch.zeros((k, d), dtype=torch.float32, device=cfg.device)
        proto_avail = torch.zeros((k,), dtype=torch.bool, device=cfg.device)
        for c, v in g_protos_torch.items():
            if v.numel() != d:
                raise ValueError("Inconsistent prototype dimensionality across classes.")
            proto_table[c] = v.view(-1)
            proto_avail[c] = True

    for ep in range(cfg.epochs):
        total_loss, total = 0.0, 0
        it = train_loader
        if progress:
            it = tqdm(train_loader, desc=f"epoch {ep+1}/{cfg.epochs}", leave=False)
        for x, y in it:
            x, y = x.to(cfg.device), y.to(cfg.device)
            opt.zero_grad()
            
            with torch.amp.autocast('cuda', enabled=use_amp):
                logits, emb = model(x)
                ce_loss = F.cross_entropy(logits, y)
                
                # Prototype Alignment Loss (FedProto Algorithm 1):
                # Compute local batch prototype per class, then distance to global prototype.
                proto_loss = torch.tensor(0.0, device=cfg.device)
                if proto_table is not None:
                    unique_classes = torch.unique(y)
                    valid_dists = []
                    for c in unique_classes:
                        c_idx = int(c.item())
                        if c_idx < proto_table.size(0) and proto_avail[c_idx]:
                            # Eq 3: Local batch prototype for class c
                            batch_proto_c = emb[y == c].mean(dim=0)
                            # Eq 8: Distance to global prototype
                            dist = torch.sum((batch_proto_c - proto_table[c_idx]) ** 2)
                            valid_dists.append(dist)
                    
                    if valid_dists:
                        proto_loss = torch.stack(valid_dists).sum()
                
                loss = ce_loss + lambda_p * proto_loss
            
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            
            total_loss += float(loss.item()) * x.size(0)
            total += x.size(0)
    return total_loss / total if total > 0 else 0

@torch.no_grad()
def compute_prototypes(model, loader: DataLoader, device: str, num_classes: int = 10) -> dict:
    """Computes class-wise mean embeddings for a client."""
    model.to(device)
    model.eval()
    
    sums = {c: None for c in range(num_classes)}
    counts = {c: 0 for c in range(num_classes)}

    for x, y in loader:
        x = x.to(device)
        _, emb = model(x)
        emb_np = emb.detach().cpu().numpy()
        y_np = y.numpy()

        for i in range(len(y_np)):
            c = int(y_np[i])
            if sums[c] is None:
                sums[c] = emb_np[i].copy()
            else:
                sums[c] += emb_np[i]
            counts[c] += 1

    prototypes = {}
    class_counts = {}
    for c in range(num_classes):
        if counts[c] > 0:
            prototypes[c] = (sums[c] / counts[c]).astype(np.float32)
            class_counts[c] = counts[c]
    return prototypes, class_counts

def evaluate_accuracy(model, test_loader: DataLoader, device: str) -> float:
    model.to(device)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += x.size(0)
    return correct / total

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
