from __future__ import annotations
from dataclasses import dataclass
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

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

    for ep in range(cfg.epochs):
        total_loss = 0.0
        total = 0
        for x, y in train_loader:
            x, y = x.to(cfg.device), y.to(cfg.device)
            opt.zero_grad()
            logits, _ = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * x.size(0)
            total += x.size(0)
        print(f"  epoch {ep+1}/{cfg.epochs} loss={total_loss/total:.4f}")

def accuracy(model, test_loader: DataLoader, device: str) -> float:
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += x.size(0)
    return correct / total