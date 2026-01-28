import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import csv


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
            nn.Conv2d(128, 1, 4, 1, 0),
        )

    def forward(self, x):
        out = self.net(x)
        return out.view(x.size(0), -1).mean(dim=1)  # Patch → scalar


# ---------------------------
# Train Discriminator
# ---------------------------
def train_discriminator(real_dir, fake_dir, epochs=30, batch_size=16):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )

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
    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )

    results = []  # (filename, score)

    D.eval()
    with torch.no_grad():
        for img_path in sorted(glob.glob(os.path.join(image_dir, "*"))):
            if not img_path.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            img = Image.open(img_path).convert("RGB")
            img = transform(img).unsqueeze(0).to(device)
            score = torch.sigmoid(D(img)).item()

            fname = os.path.basename(img_path)
            results.append((fname, score))

    return results


def save_scores(results, out_path):
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "d_score"])
        for r in results:
            writer.writerow(r)


def save_evaluation_figure(real_scores, fake_scores, out_path):
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # ---- Histogram ----
    axs[0, 0].hist(real_scores, bins=20, alpha=0.7, label="Real", color="green")
    axs[0, 0].hist(fake_scores, bins=20, alpha=0.7, label="Fake", color="red")
    axs[0, 0].set_title("Discriminator Score Distribution")
    axs[0, 0].set_xlabel("D-score")
    axs[0, 0].set_ylabel("Frequency")
    axs[0, 0].legend()

    # ---- Boxplot ----
    axs[0, 1].boxplot(
        [real_scores, fake_scores], labels=["Real", "Fake"], showfliers=True
    )
    axs[0, 1].set_title("D-score Boxplot Comparison")
    axs[0, 1].set_ylabel("D-score")

    # ---- Per-image scatter ----
    axs[1, 0].scatter(range(len(real_scores)), real_scores, alpha=0.7, label="Real")
    axs[1, 0].scatter(range(len(fake_scores)), fake_scores, alpha=0.7, label="Fake")
    axs[1, 0].set_title("Per-image D-scores")
    axs[1, 0].set_xlabel("Image Index")
    axs[1, 0].set_ylabel("D-score")
    axs[1, 0].legend()

    # ---- Summary text ----
    gap = real_scores.mean() - fake_scores.mean()
    text = (
        "EVALUATION SUMMARY\n"
        "=========================\n\n"
        f"Real mean D-score : {real_scores.mean():.4f} ± {real_scores.std():.4f}\n"
        f"Fake mean D-score : {fake_scores.mean():.4f} ± {fake_scores.std():.4f}\n"
        f"Mean gap (R − F)  : {gap:.4f}\n\n"
        "Interpretation:\n"
        "- Higher real score = discriminator confidence\n"
        "- Smaller gap = better inpainting realism\n"
    )

    axs[1, 1].axis("off")
    axs[1, 1].text(0.05, 0.95, text, va="top", ha="left", fontsize=11)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    real_dir = "real_backgrounds"
    fake_dir = "inpainted_backgrounds"

    # ---- train ----
    D, device = train_discriminator(real_dir, fake_dir)

    # ---- evaluate ----
    real_results = evaluate_images(D, real_dir, device)
    fake_results = evaluate_images(D, fake_dir, device)

    real_scores = np.array([s for _, s in real_results])
    fake_scores = np.array([s for _, s in fake_results])

    # ---- create run folder ----
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join("results", f"run_{run_id}")
    os.makedirs(out_dir, exist_ok=True)

    # ---- save data ----
    save_scores(real_results, os.path.join(out_dir, "real_scores.csv"))
    save_scores(fake_results, os.path.join(out_dir, "fake_scores.csv"))

    torch.save(D.state_dict(), os.path.join(out_dir, "discriminator_only.pth"))

    # ---- summary ----
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write("DISCRIMINATOR EVALUATION SUMMARY\n")
        f.write("=" * 35 + "\n\n")
        f.write(
            f"Real mean D-score: {real_scores.mean():.4f} ± {real_scores.std():.4f}\n"
        )
        f.write(
            f"Fake mean D-score: {fake_scores.mean():.4f} ± {fake_scores.std():.4f}\n"
        )
        f.write(
            f"Mean gap (Real - Fake): {(real_scores.mean() - fake_scores.mean()):.4f}\n"
        )

    save_evaluation_figure(
        real_scores, fake_scores, os.path.join(out_dir, "evaluation_figure.png")
    )

    # ---- console output ----
    print("\nEVALUATION RESULTS")
    print("------------------")
    print(f"Real mean D-score: {real_scores.mean():.3f} ± {real_scores.std():.3f}")
    print(f"Fake mean D-score: {fake_scores.mean():.3f} ± {fake_scores.std():.3f}")
    print(f"Results saved to: {out_dir}")
