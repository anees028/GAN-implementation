import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


# Custom Dataset for loading images from folder
class CustomImageDataset(Dataset):
    def __init__(self, image_folder, transform=None, img_size=64):
        self.image_folder = image_folder
        self.transform = transform
        self.img_size = img_size

        # Support multiple image formats
        self.image_paths = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif"]:
            self.image_paths.extend(glob.glob(os.path.join(image_folder, ext)))
            self.image_paths.extend(glob.glob(os.path.join(image_folder, ext.upper())))

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {image_folder}")

        print(f"Found {len(self.image_paths)} images in {image_folder}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, 0  # Return 0 as dummy label


# Generator Network - Dynamic size
class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_channels=3, img_size=64):
        super(Generator, self).__init__()
        self.img_channels = img_channels
        self.img_size = img_size
        self.init_size = img_size // 4

        self.fc = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size**2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, img_channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.fc(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


# Discriminator Network - Dynamic size
class Discriminator(nn.Module):
    def __init__(self, img_channels=3, img_size=64):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [
                nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25),
            ]
            if bn:
                block.append(nn.BatchNorm2d(out_filters))
            return block

        self.model = nn.Sequential(
            *discriminator_block(img_channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # Calculate size after convolutions
        ds_size = img_size // 2**4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size**2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity


# Training Function
def train_gan(
    image_folder,
    epochs=100,
    batch_size=32,
    latent_dim=100,
    lr=0.0002,
    img_size=64,
    img_channels=3,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Transforms
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * img_channels, [0.5] * img_channels),
        ]
    )

    # Load custom dataset
    dataset = CustomImageDataset(image_folder, transform=transform, img_size=img_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # Initialize models
    generator = Generator(latent_dim, img_channels, img_size).to(device)
    discriminator = Discriminator(img_channels, img_size).to(device)

    # Loss function
    adversarial_loss = nn.BCELoss()

    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    # Training metrics
    g_losses = []
    d_losses = []

    print("\nStarting Training...")
    print("-" * 50)

    for epoch in range(epochs):
        epoch_g_loss = 0
        epoch_d_loss = 0

        for i, (imgs, _) in enumerate(dataloader):
            batch_size_current = imgs.size(0)

            # Adversarial ground truths
            valid = torch.ones(batch_size_current, 1).to(device)
            fake = torch.zeros(batch_size_current, 1).to(device)

            real_imgs = imgs.to(device)

            # -----------------
            # Train Generator
            # -----------------
            optimizer_G.zero_grad()

            # Sample noise
            z = torch.randn(batch_size_current, latent_dim).to(device)

            # Generate fake images
            gen_imgs = generator(z)

            # Generator loss
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            # Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Discriminator loss on real images
            real_loss = adversarial_loss(discriminator(real_imgs), valid)

            # Discriminator loss on fake images
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)

            # Total discriminator loss
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()

        # Record average losses
        avg_g_loss = epoch_g_loss / len(dataloader)
        avg_d_loss = epoch_d_loss / len(dataloader)
        g_losses.append(avg_g_loss)
        d_losses.append(avg_d_loss)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch+1}/{epochs}] | D Loss: {avg_d_loss:.4f} | G Loss: {avg_g_loss:.4f}"
            )

    print("-" * 50)
    print("Training Complete!")

    return generator, discriminator, g_losses, d_losses, device


# Evaluation Functions
def evaluate_gan(
    generator,
    discriminator,
    image_folder,
    device,
    latent_dim=100,
    n_samples=500,
    img_size=64,
    img_channels=3,
):
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)

    generator.eval()
    discriminator.eval()

    with torch.no_grad():
        # Generate fake images
        z = torch.randn(n_samples, latent_dim).to(device)
        fake_imgs = generator(z)

        # Get discriminator predictions
        fake_preds = discriminator(fake_imgs).cpu().numpy()

        # Load real images
        transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * img_channels, [0.5] * img_channels),
            ]
        )
        dataset = CustomImageDataset(
            image_folder, transform=transform, img_size=img_size
        )
        n_real = min(n_samples, len(dataset))
        real_imgs = torch.stack([dataset[i][0] for i in range(n_real)]).to(device)
        real_preds = discriminator(real_imgs).cpu().numpy()

        # Calculate metrics
        print("\n1. DISCRIMINATOR PERFORMANCE:")
        print(
            f"   Real images classified as real: {(real_preds > 0.5).mean()*100:.2f}%"
        )
        print(
            f"   Fake images classified as fake: {(fake_preds <= 0.5).mean()*100:.2f}%"
        )
        print(
            f"   Overall accuracy: {((real_preds > 0.5).mean() + (fake_preds <= 0.5).mean())/2*100:.2f}%"
        )

        print("\n2. GENERATOR PERFORMANCE:")
        print(
            f"   Fake images fooling discriminator: {(fake_preds > 0.5).mean()*100:.2f}%"
        )
        print(f"   Average discriminator confidence on fakes: {fake_preds.mean():.4f}")

        print("\n3. DISTRIBUTION STATISTICS:")
        print(
            f"   Real images mean prediction: {real_preds.mean():.4f} ± {real_preds.std():.4f}"
        )
        print(
            f"   Fake images mean prediction: {fake_preds.mean():.4f} ± {fake_preds.std():.4f}"
        )

    print("=" * 50)

    return fake_imgs, real_imgs, fake_preds, real_preds


# Visualization Functions
def visualize_results(
    generator,
    g_losses,
    d_losses,
    image_folder,
    device,
    latent_dim=100,
    img_size=64,
    img_channels=3,
):
    fig = plt.figure(figsize=(15, 10))

    # Plot 1: Training Loss
    plt.subplot(2, 3, 1)
    plt.plot(g_losses, label="Generator Loss", linewidth=2)
    plt.plot(d_losses, label="Discriminator Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Time")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Generated Images
    with torch.no_grad():
        z = torch.randn(16, latent_dim).to(device)
        gen_imgs = generator(z).cpu()

    plt.subplot(2, 3, 2)
    grid = np.zeros((4 * img_size, 4 * img_size, img_channels))
    for i in range(4):
        for j in range(4):
            img = gen_imgs[i * 4 + j].permute(1, 2, 0).numpy()
            img = (img + 1) / 2  # Denormalize
            grid[
                i * img_size : (i + 1) * img_size, j * img_size : (j + 1) * img_size
            ] = img
    plt.imshow(np.clip(grid, 0, 1))
    plt.title("Generated Images (16 samples)")
    plt.axis("off")

    # Plot 3: Real Images (for comparison)
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * img_channels, [0.5] * img_channels),
        ]
    )
    dataset = CustomImageDataset(image_folder, transform=transform, img_size=img_size)

    plt.subplot(2, 3, 3)
    grid_real = np.zeros((4 * img_size, 4 * img_size, img_channels))
    for i in range(4):
        for j in range(4):
            idx = min(i * 4 + j, len(dataset) - 1)
            img = dataset[idx][0].permute(1, 2, 0).numpy()
            img = (img + 1) / 2  # Denormalize
            grid_real[
                i * img_size : (i + 1) * img_size, j * img_size : (j + 1) * img_size
            ] = img
    plt.imshow(np.clip(grid_real, 0, 1))
    plt.title("Real Images (16 samples)")
    plt.axis("off")

    # Plot 4: Loss difference
    plt.subplot(2, 3, 4)
    loss_diff = np.array(g_losses) - np.array(d_losses)
    plt.plot(loss_diff, color="purple", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss Difference (G - D)")
    plt.title("Generator vs Discriminator Loss Gap")
    plt.axhline(y=0, color="r", linestyle="--", alpha=0.5)
    plt.grid(True, alpha=0.3)

    # Plot 5: Sample diversity
    plt.subplot(2, 3, 5)
    with torch.no_grad():
        z_diverse = torch.randn(9, latent_dim).to(device)
        diverse_imgs = generator(z_diverse).cpu()

    grid_diverse = np.zeros((3 * img_size, 3 * img_size, img_channels))
    for i in range(3):
        for j in range(3):
            img = diverse_imgs[i * 3 + j].permute(1, 2, 0).numpy()
            img = (img + 1) / 2  # Denormalize
            grid_diverse[
                i * img_size : (i + 1) * img_size, j * img_size : (j + 1) * img_size
            ] = img
    plt.imshow(np.clip(grid_diverse, 0, 1))
    plt.title("Generated Diversity Test")
    plt.axis("off")

    # Plot 6: Summary
    plt.subplot(2, 3, 6)
    plt.axis("off")
    summary_text = f"""
    TRAINING SUMMARY
    {'='*30}
    
    Final Generator Loss: {g_losses[-1]:.4f}
    Final Discriminator Loss: {d_losses[-1]:.4f}
    
    Total Epochs: {len(g_losses)}
    Image Size: {img_size}x{img_size}
    Channels: {img_channels}
    
    Best G Loss: {min(g_losses):.4f}
    Best D Loss: {min(d_losses):.4f}
    
    Average G Loss: {np.mean(g_losses):.4f}
    Average D Loss: {np.mean(d_losses):.4f}
    """
    plt.text(
        0.1,
        0.5,
        summary_text,
        fontsize=10,
        family="monospace",
        verticalalignment="center",
    )

    plt.tight_layout()
    plt.savefig("gan_results.png", dpi=150, bbox_inches="tight")
    print("\n✓ Results saved to 'gan_results.png'")
    plt.show()


# Main execution
if __name__ == "__main__":
    print("GAN Training and Evaluation Pipeline")
    print("=" * 50)

    # Configuration
    IMAGE_FOLDER = "./training_images"  # Change this to your image folder
    IMG_SIZE = 128  # Image size (will resize all images to this)
    IMG_CHANNELS = 3  # 3 for RGB, 1 for grayscale
    EPOCHS = 100
    BATCH_SIZE = 32
    LATENT_DIM = 100
    LEARNING_RATE = 0.0002

    # Create folder if it doesn't exist
    if not os.path.exists(IMAGE_FOLDER):
        os.makedirs(IMAGE_FOLDER)
        print(f"\n⚠ Created folder: {IMAGE_FOLDER}")
        print(f"Please add your training images to this folder and run again.")
        exit()

    # Train the GAN
    generator, discriminator, g_losses, d_losses, device = train_gan(
        image_folder=IMAGE_FOLDER,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        latent_dim=LATENT_DIM,
        lr=LEARNING_RATE,
        img_size=IMG_SIZE,
        img_channels=IMG_CHANNELS,
    )

    # Evaluate the GAN
    fake_imgs, real_imgs, fake_preds, real_preds = evaluate_gan(
        generator,
        discriminator,
        IMAGE_FOLDER,
        device,
        latent_dim=LATENT_DIM,
        n_samples=500,
        img_size=IMG_SIZE,
        img_channels=IMG_CHANNELS,
    )

    # Visualize results
    visualize_results(
        generator,
        g_losses,
        d_losses,
        IMAGE_FOLDER,
        device,
        latent_dim=LATENT_DIM,
        img_size=IMG_SIZE,
        img_channels=IMG_CHANNELS,
    )

    # Save models
    torch.save(generator.state_dict(), "generator.pth")
    torch.save(discriminator.state_dict(), "discriminator.pth")
    print("\n✓ Models saved as 'generator.pth' and 'discriminator.pth'")
    print("✓ All operations completed successfully!")
