import torch
import torch.nn as nn
import torch.optim as optim

from helper_lib.data_loader import get_mnist_gan_loader
from helper_lib.model import get_model
from helper_lib.trainer import train_gan
from helper_lib.generator import generate_samples

device = torch.device('mps') if torch.backends.mps.is_available() else \
         torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device: {device}")

loader = get_mnist_gan_loader(batch_size=128)
model = get_model("GAN")
criterion = nn.BCELoss()
optimizer = {
    "generator": optim.Adam(model.generator.parameters(), lr=2e-4, betas=(0.5, 0.999)),
    "discriminator": optim.Adam(model.discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999)),
}

model = train_gan(model, loader, criterion, optimizer, device=device, epochs=20)

torch.save(model.generator.state_dict(), "app/gan_generator.pth")
print("Saved app/gan_generator.pth")

generate_samples(model, device, num_samples=10)