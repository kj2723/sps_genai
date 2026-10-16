import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

from app.cnn_model import CNN, TRANSFORM, WEIGHTS_PATH

EPOCHS = 5
BATCH_SIZE = 64
LR = 1e-3


class CIFAR10(Dataset):
    """CIFAR-10 loaded from Hugging Face, transformed for the 64x64 CNN."""

    def __init__(self, split: str):
        self.ds = load_dataset("uoft-cs/cifar10", split=split)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        row = self.ds[i]
        return TRANSFORM(row["img"].convert("RGB")), row["label"]


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            correct += (model(images).argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return correct / total


def main():
    device = get_device()
    print(f"Device: {device}")

    train_set = CIFAR10("train")
    test_set = CIFAR10("test")
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=256)

    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch + 1}/{EPOCHS}: loss={total_loss / len(train_loader):.4f}, test acc={acc:.3f}")

    model.to("cpu")
    torch.save(model.state_dict(), WEIGHTS_PATH)
    print(f"Saved weights to {WEIGHTS_PATH}")


if __name__ == "__main__":
    main()