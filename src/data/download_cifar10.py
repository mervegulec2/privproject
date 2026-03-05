from torchvision import datasets, transforms

def main():
    transform = transforms.Compose([transforms.ToTensor()])

    datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
    datasets.CIFAR10(root="data", train=False, download=True, transform=transform)

    print("✅ CIFAR-10 downloaded into ./data")

if __name__ == "__main__":
    main()