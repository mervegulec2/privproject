import torch
import torch.nn as nn
from torchvision.models import resnet18

class ResNet18Cifar(nn.Module):
    """
    ResNet-18 adapted for CIFAR-10:
    - 3x3 conv1 with stride 1
    - Identity maxpool
    - Global average pooling followed by FC
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        m = resnet18(weights=None)
        m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        m.maxpool = nn.Identity()

        self.backbone = nn.Sequential(
            m.conv1, m.bn1, m.relu, m.maxpool,
            m.layer1, m.layer2, m.layer3, m.layer4,
        )
        self.avgpool = m.avgpool
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        feat = self.backbone(x)
        pooled = self.avgpool(feat)
        emb = torch.flatten(pooled, 1)
        logits = self.fc(emb)
        return logits, emb