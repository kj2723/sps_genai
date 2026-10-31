from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_mnist_gan_loader(data_dir="data", batch_size=128):
    # Scale pixels to [-1, 1] to match the generator's Tanh output
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)