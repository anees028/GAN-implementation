### 3. Set up the Python environment

```bash
python -m venv .venv     # For Windows
python3 -m venv .venv     # For Mac

.venv\Scripts\activate           # On Windows

source .venv/bin/activate           # For Mac

pip install -r requirements.txt
```


# GAN Evaluation Project

This project uses a Discriminator model to evaluate the realness of synthetic images by comparing them to real images. The Discriminator is trained to distinguish between real and fake images, and the DScore is computed for each image.

## Setup

1. Install dependencies:

```bash
   pip install -r requirements.txt

  python3 train_discriminator.py

  python3 evaluate_dscore.py
```