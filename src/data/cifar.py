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
    train_ds = datasets.CIFAR10(root=cfg.root, train=True, download=False, transform=train_tf)
    test_ds  = datasets.CIFAR10(root=cfg.root, train=False, download=False, transform=test_tf)
    return train_ds, test_ds