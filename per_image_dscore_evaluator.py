import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import csv

# -----------------------------
# Discriminator (same as trained)
# -----------------------------
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


# -----------------------------
# Load model
# -----------------------------
def load_discriminator(model_path, device):
    model = Discriminator().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


# -----------------------------
# Image transform
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])


def load_image(path, device):
    img = Image.open(path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)
    return img


# -----------------------------
# Per-image evaluation
# -----------------------------
def evaluate_per_image(real_dir, fake_dir, model_path, output_dir="results"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    D = load_discriminator(model_path, device)

    os.makedirs(output_dir, exist_ok=True)
    per_image_dir = os.path.join(output_dir, "per_image_scores")
    os.makedirs(per_image_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "scores_per_image.csv")

    rows = []
    real_scores = []
    fake_scores = []

    common_files = sorted(set(os.listdir(real_dir)) & set(os.listdir(fake_dir)))

    with torch.no_grad():
        for fname in common_files:
            real_img = load_image(os.path.join(real_dir, fname), device)
            fake_img = load_image(os.path.join(fake_dir, fname), device)

            real_score = torch.sigmoid(D(real_img)).item()
            fake_score = torch.sigmoid(D(fake_img)).item()
            delta = real_score - fake_score

            real_scores.append(real_score)
            fake_scores.append(fake_score)

            rows.append([fname, real_score, fake_score, delta])

            # Save individual report
            with open(os.path.join(per_image_dir, fname.replace(".", "_") + ".txt"), "w") as f:
                f.write(f"Image: {fname}\n")
                f.write(f"Real D-score: {real_score:.4f}\n")
                f.write(f"Fake D-score: {fake_score:.4f}\n")
                f.write(f"Delta (Real - Fake): {delta:.4f}\n")

    # Save CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "real_dscore", "fake_dscore", "delta"])
        writer.writerows(rows)

    # Save summary
    with open(os.path.join(output_dir, "summary.txt"), "w") as f:
        f.write("DISCRIMINATOR PER-IMAGE EVALUATION SUMMARY\n")
        f.write("="*45 + "\n")
        f.write(f"Total image pairs: {len(rows)}\n\n")
        f.write(f"Real mean D-score: {np.mean(real_scores):.4f}\n")
        f.write(f"Fake mean D-score: {np.mean(fake_scores):.4f}\n")
        f.write(f"Mean ΔD-score: {np.mean(np.array(real_scores) - np.array(fake_scores)):.4f}\n")

    print("✓ Per-image evaluation complete")
    print(f"✓ Results saved in '{output_dir}/'")


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    evaluate_per_image(
        real_dir="data/real_backgrounds",
        fake_dir="data/inpainted_backgrounds",
        model_path="discriminator_only.pth",
        output_dir="results"
    )
