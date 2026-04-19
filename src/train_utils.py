import torch
import torch.nn.functional as F
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

def train_one_client(model, train_loader: DataLoader, cfg: TrainConfig):
    model.to(cfg.device)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    
    # Mixed precision setup
    use_amp = (cfg.device == "cuda")
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    for ep in range(cfg.epochs):
        total_loss, total = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(cfg.device), y.to(cfg.device)
            opt.zero_grad()
            
            with torch.amp.autocast('cuda', enabled=use_amp):
                logits, _ = model(x)
                loss = F.cross_entropy(logits, y)
            
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            
            total_loss += float(loss.item()) * x.size(0)
            total += x.size(0)
        print(f"  Epoch {ep+1}/{cfg.epochs} - Loss: {total_loss/total:.4f}")

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
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
