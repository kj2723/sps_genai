import torch
from tqdm import tqdm


def train_gan(model, data_loader, criterion, optimizer, device='cpu', epochs=10):
    """optimizer is a dict: {'generator': opt_G, 'discriminator': opt_D}"""
    gen = model.generator.to(device)
    disc = model.discriminator.to(device)
    opt_G = optimizer["generator"]
    opt_D = optimizer["discriminator"]

    for epoch in range(epochs):
        loop = tqdm(data_loader, ncols=120, desc=f"Epoch {epoch+1}/{epochs}")
        for real, _ in loop:
            real = real.to(device)
            b = real.size(0)
            ones = torch.ones(b, 1, device=device)
            zeros = torch.zeros(b, 1, device=device)

            # Train discriminator: real -> 1, fake -> 0
            fake = gen(torch.randn(b, model.z_dim, device=device))
            loss_D = criterion(disc(real), ones) + criterion(disc(fake.detach()), zeros)
            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # Train generator: make discriminator output 1 on fakes
            loss_G = criterion(disc(fake), ones)
            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            loop.set_postfix({"D loss": f"{loss_D.item():.4f}", "G loss": f"{loss_G.item():.4f}"})
    return model