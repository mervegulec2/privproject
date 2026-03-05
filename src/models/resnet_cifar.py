from __future__ import annotations
import torch
import torch.nn as nn
from torchvision.models import resnet18

class ResNet18Cifar(nn.Module):
    """
    CIFAR-10 için ResNet-18:
    - conv1: 3x3, stride 1
    - maxpool kaldırılır
    - embedding: avgpool sonrası 512-d
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        m = resnet18(weights=None)

        # CIFAR fix
        m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        m.maxpool = nn.Identity()

        self.backbone = nn.Sequential(
            m.conv1, m.bn1, m.relu, m.maxpool,
            m.layer1, m.layer2, m.layer3, m.layer4,
        )
        self.avgpool = m.avgpool  # AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        feat = self.backbone(x)
        pooled = self.avgpool(feat)                # [B,512,1,1]
        emb = torch.flatten(pooled, 1)            # [B,512]
        logits = self.fc(emb)                     # [B,10]
        return logits, emb