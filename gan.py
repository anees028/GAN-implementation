import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

# ---------------------------
# Dataset
# ---------------------------
class RealFakeDataset(Dataset):
    def __init__(self, real_dir, fake_dir, transform=None):
        self.samples = []
        self.transform = transform

        for p in glob.glob(os.path.join(real_dir, "*")):
            self.samples.append((p, 1))  # real = 1

        for p in glob.glob(os.path.join(fake_dir, "*")):
            self.samples.append((p, 0))  # fake = 0

        if len(self.samples) == 0:
            raise RuntimeError("No images found.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)


# ---------------------------
# Discriminator (Patch-style)
# ---------------------------
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 1, 4, 1, 0)
        )

    def forward(self, x):
        out = self.net(x)
        return out.view(x.size(0), -1).mean(dim=1)  # Patch → scalar


# ---------------------------
# Train Discriminator
# ---------------------------
def train_discriminator(real_dir, fake_dir, epochs=30, batch_size=16):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    dataset = RealFakeDataset(real_dir, fake_dir, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    D = Discriminator().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(D.parameters(), lr=2e-4)

    for epoch in range(epochs):
        losses = []

        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            preds = D(imgs)
            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        print(f"Epoch {epoch+1}/{epochs} | D Loss: {np.mean(losses):.4f}")

    torch.save(D.state_dict(), "discriminator_only.pth")
    print("✓ Discriminator trained & saved")

    return D, device


# ---------------------------
# Evaluation
# ---------------------------
def evaluate_images(D, image_dir, device):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    scores = []

    D.eval()
    with torch.no_grad():
        for img_path in glob.glob(os.path.join(image_dir, "*")):
            img = Image.open(img_path).convert("RGB")
            img = transform(img).unsqueeze(0).to(device)
            score = torch.sigmoid(D(img)).item()
            scores.append(score)

    return np.array(scores)


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    real_dir = "real_backgrounds"
    fake_dir = "inpainted_backgrounds"

    D, device = train_discriminator(real_dir, fake_dir)

    real_scores = evaluate_images(D, real_dir, device)
    fake_scores = evaluate_images(D, fake_dir, device)

    print("\nEVALUATION RESULTS")
    print("------------------")
    print(f"Real mean D-score: {real_scores.mean():.3f} ± {real_scores.std():.3f}")
    print(f"Fake mean D-score: {fake_scores.mean():.3f} ± {fake_scores.std():.3f}")
