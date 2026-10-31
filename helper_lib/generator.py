import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def generate_samples(model, device, num_samples=10):
    gen = model.generator.to(device)
    gen.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, model.z_dim, device=device)
        imgs = gen(z).cpu()
    grid = make_grid(imgs, nrow=5, normalize=True)
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.show()