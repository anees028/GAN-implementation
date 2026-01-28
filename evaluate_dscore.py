import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from discriminator_model import build_discriminator
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


# Load the trained Discriminator
discriminator = build_discriminator()
discriminator.load_weights("path_to_trained_discriminator_weights")

# Load real and synthetic images
real_images = load_and_preprocess_images("./input_images")
synthetic_images = load_and_preprocess_images("./generated_images")

# Evaluate the DScore
real_scores = discriminator.predict(real_images)
synthetic_scores = discriminator.predict(synthetic_images)

# DScore is the output of the discriminator (closer to 1 means real, closer to 0 means fake)
d_scores_real = np.squeeze(real_scores)
d_scores_fake = np.squeeze(synthetic_scores)

# Combine real and fake DScores for plotting
d_scores = np.concatenate([d_scores_real, d_scores_fake])
labels = ["Real"] * len(d_scores_real) + ["Synthetic"] * len(d_scores_fake)

# Plot the DScore comparison using a bar chart
plt.figure(figsize=(10, 6))
plt.bar(
    range(len(d_scores)),
    d_scores,
    color=["green"] * len(d_scores_real) + ["red"] * len(d_scores_fake),
)
plt.xlabel("Image Index")
plt.ylabel("DScore")
plt.title("Evaluation of Real vs Synthetic Images (DScore)")
plt.xticks(range(len(d_scores)), labels, rotation=90)
plt.show()


# This file evaluates the DScore for both real and synthetic images using the trained Discriminator. 
# It uses the trained Discriminator to output the DScore (probability that the image is real). 
# It then visualizes the scores using a bar chart.