from torchvision import datasets, transforms

def main():
    print("Downloading CIFAR-10...")
    transform = transforms.Compose([transforms.ToTensor()])
    datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
    datasets.CIFAR10(root="data", train=False, download=True, transform=transform)
    print("Done. Dataset saved in data/")

if __name__ == "__main__":
    main()