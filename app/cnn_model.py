import io
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

WEIGHTS_PATH = Path(__file__).parent / "cnn_weights.pth"

# CIFAR images are 32x32; convert to 64x64
TRANSFORM = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


class CNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),   # 64x64x16
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                  # 32x32x16
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # 32x32x32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                  # 16x16x32
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                                           # 8192
            nn.Linear(32 * 16 * 16, 100),
            nn.ReLU(),
            nn.Linear(100, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class CNNClassifier:
    def __init__(self, weights_path: Path = WEIGHTS_PATH):
        self.model = CNN()
        self.model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        self.model.eval()

    def predict(self, image_bytes: bytes) -> dict:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        x = TRANSFORM(image).unsqueeze(0)          # [1, 3, 64, 64]
        with torch.no_grad():
            probs = torch.softmax(self.model(x), dim=1)[0]
        idx = int(probs.argmax())
        return {"class": CLASSES[idx], "confidence": round(float(probs[idx]), 4)}