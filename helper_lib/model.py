import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super().__init__()
        # Fully connected layer to 7x7x128, then reshape
        self.fc = nn.Linear(z_dim, 7 * 7 * 128)
        self.net = nn.Sequential(
            # 128 -> 64, 7x7 -> 14x14
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 64 -> 1, 14x14 -> 28x28
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(self.fc(z).view(-1, 128, 7, 7))


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # 1 -> 64, 28x28 -> 14x14
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # 64 -> 128, 14x14 -> 7x7
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid(),  # real/fake probability
        )

    def forward(self, x):
        return self.net(x)


class GAN(nn.Module):
    """Holds the generator and discriminator together."""
    def __init__(self, z_dim=100):
        super().__init__()
        self.z_dim = z_dim
        self.generator = Generator(z_dim)
        self.discriminator = Discriminator()


def get_model(model_name):
    # Keep any existing FCNN / CNN / EnhancedCNN / VAE branches above this one
    if model_name == "GAN":
        model = GAN()
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model