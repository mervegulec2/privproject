import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim.swa_utils import update_bn

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
    # mixup_alpha > 0 enables batch mixup (prototype term omitted on mixed batches)
    mixup_alpha: float = 0.0
    # Average last swa_last_epochs epoch weights into the final model (then optional BN refresh)
    swa_enabled: bool = False
    swa_last_epochs: int = 2


def _mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float):
    if alpha <= 0.0:
        return None
    lam = float(np.random.beta(alpha, alpha))
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1.0 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def train_local_proto(
    model,
    train_loader: DataLoader,
    global_protos: dict,
    cfg: TrainConfig,
    lambda_p: float = 0.05,
    progress: bool = False,
    cid: int = -1,
    class_weights: torch.Tensor = None,
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
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

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

    swa_begin = max(0, cfg.epochs - cfg.swa_last_epochs) if cfg.swa_enabled else cfg.epochs
    swa_snapshots: list[dict] = []

    for ep in range(cfg.epochs):
        total_loss, total = 0.0, 0
        it = train_loader
        if progress:
            it = tqdm(train_loader, desc=f"epoch {ep+1}/{cfg.epochs}", leave=False)
        for x, y in it:
            x, y = x.to(cfg.device), y.to(cfg.device)
            opt.zero_grad()

            mix = _mixup_data(x, y, cfg.mixup_alpha) if cfg.mixup_alpha > 0 else None
            use_proto = proto_table is not None and mix is None

            with torch.cuda.amp.autocast(enabled=use_amp):
                if mix is not None:
                    mixed_x, y_a, y_b, lam = mix
                    logits, emb = model(mixed_x)
                    ce_loss = lam * F.cross_entropy(logits, y_a) + (1.0 - lam) * F.cross_entropy(
                        logits, y_b
                    )
                    proto_loss = torch.tensor(0.0, device=cfg.device)
                else:
                    logits, emb = model(x)
                    if class_weights is not None:
                        ce_loss = F.cross_entropy(logits, y, weight=class_weights)
                    else:
                        ce_loss = F.cross_entropy(logits, y)

                    proto_loss = torch.tensor(0.0, device=cfg.device)
                    if use_proto:
                        unique_classes = torch.unique(y)
                        valid_dists = []
                        for c in unique_classes:
                            c_idx = int(c.item())
                            if c_idx < proto_table.size(0) and proto_avail[c_idx]:
                                batch_proto_c = emb[y == c].mean(dim=0)
                                dist = torch.sum((batch_proto_c - proto_table[c_idx]) ** 2)
                                valid_dists.append(dist)

                        if valid_dists:
                            proto_loss = torch.stack(valid_dists).sum()

                loss = ce_loss + lambda_p * proto_loss

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            bs = x.size(0)
            total_loss += float(loss.item()) * bs
            total += bs

        epoch_loss = total_loss / total if total > 0 else 0
        if not progress:
            print(f"[Client {cid}] Epoch {ep+1}/{cfg.epochs} - Loss: {epoch_loss:.4f}", flush=True)

        if cfg.swa_enabled and ep >= swa_begin:
            swa_snapshots.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})

    if swa_snapshots:
        avg_sd: dict[str, torch.Tensor] = {}
        for k in swa_snapshots[0]:
            t0 = swa_snapshots[0][k]
            if t0.dtype in (torch.long, torch.int32, torch.bool):
                avg_sd[k] = swa_snapshots[-1][k].clone().to(device=cfg.device)
            else:
                stacked = torch.stack([snap[k].float() for snap in swa_snapshots])
                avg_sd[k] = stacked.mean(dim=0).to(device=cfg.device, dtype=t0.dtype)
        model.load_state_dict(avg_sd)
        model.train()
        try:
            update_bn(train_loader, model, device=torch.device(cfg.device))
        except Exception:
            model.eval()
        else:
            model.eval()

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

def compute_class_weights(dataset, indices, num_classes=10, device="cpu"):
    """Computes inverse frequency class weights to balance heavily skewed local data."""
    if hasattr(dataset, "targets"):
        targets = np.array(dataset.targets)
    else:
        # Fallback for datasets without .targets array exposed
        targets = np.array([dataset[i][1] for i in range(len(dataset))])
        
    local_targets = targets[indices]
    classes, counts = np.unique(local_targets, return_counts=True)
    
    weights = np.zeros(num_classes, dtype=np.float32)
    total_samples = len(local_targets)
    
    # Use len(classes) so the expected weight is 1.0 for the classes actually present
    for c, count in zip(classes, counts):
        if count > 0:
            weights[c] = total_samples / (len(classes) * count)
            
    return torch.tensor(weights, dtype=torch.float32, device=device)

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