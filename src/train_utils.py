import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from torch.utils.data import DataLoader

@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 5
    batch_size: int = 64
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    device: str = "cpu"


def train_local_proto(model, train_loader: DataLoader, global_protos: dict, cfg: TrainConfig, lambda_p: float = 0.05):
    """Collaborative training with Prototype Alignment Loss."""
    model.to(cfg.device)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    
    use_amp = (cfg.device == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Convert global prototypes to torch tensors on correct device
    g_protos_torch = {c: torch.tensor(v).to(cfg.device) for c, v in global_protos.items()}

    for ep in range(cfg.epochs):
        total_loss, total = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(cfg.device), y.to(cfg.device)
            opt.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits, emb = model(x)
                ce_loss = F.cross_entropy(logits, y)
                
                # Prototype Alignment Loss
                proto_loss = 0.0
                matched_samples = 0
                for c in g_protos_torch.keys():
                    mask = (y == c)
                    if mask.any():
                        sample_embs = emb[mask]
                        # Squared L2 distance
                        proto_loss += torch.sum((sample_embs - g_protos_torch[c])**2)
                        matched_samples += sample_embs.size(0)
                
                if matched_samples > 0:
                    proto_loss = proto_loss / matched_samples
                
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
    for c in range(num_classes):
        if counts[c] > 0:
            prototypes[c] = (sums[c] / counts[c]).astype(np.float32)
    return prototypes

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
