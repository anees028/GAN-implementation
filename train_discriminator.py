import numpy as np
import tensorflow as tf
from discriminator_model import build_discriminator
from tensorflow.keras.preprocessing import image
import os


def load_and_preprocess_images(folder, target_size=(2560, 1440)):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if os.path.isfile(img_path) and filename.lower().endswith(
            (".png", ".jpg", ".jpeg")
        ):
            # Load and resize the image to the new target size (2560x1440)
            img = image.load_img(img_path, target_size=target_size)

            # Convert image to numpy array and normalize it to [-1, 1]
            img_array = image.img_to_array(img)
            img_array = img_array / 127.5 - 1.0  # Normalize to [-1, 1]

            images.append(img_array)  # Append without expanding dimensions

    # Return the images as a numpy array with shape (batch_size, 2560, 1440, 3)
    return np.array(images)


# Load and preprocess real and synthetic images (now resized to 2560x1440)
real_images = load_and_preprocess_images("./input_images", target_size=(2560, 1440))
synthetic_images = load_and_preprocess_images(
    "./generated_images", target_size=(2560, 1440)
)

# Labels for real (1) and synthetic (0) images
real_labels = np.ones((real_images.shape[0], 1))
fake_labels = np.zeros((synthetic_images.shape[0], 1))

# Create the Discriminator model
discriminator = build_discriminator(input_shape=(2560, 1440, 3))
discriminator.compile(
    optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
)

# Train the Discriminator
discriminator.fit(
    np.concatenate([real_images, synthetic_images]),
    np.concatenate([real_labels, fake_labels]),
    epochs=5,
    batch_size=32,
)


# This file contains the code to train the Discriminator on both real and synthetic images.
# It loads the real and synthetic images, prepares the labels (1 for real, 0 for fake), and trains the Discriminator.
