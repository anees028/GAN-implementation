import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from gan_model import build_generator, build_discriminator, build_gan

# Define the folder to save generated images
save_dir = "./generated_images"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Define the input folder containing the real images
input_folder = "./input_images"  # Folder where real images are stored


# Function to load and preprocess images
def load_images_from_folder(folder, target_size=(64, 64)):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if os.path.isfile(img_path) and filename.lower().endswith(
            (".png", ".jpg", ".jpeg")
        ):
            img = image.load_img(img_path, target_size=target_size)
            img_array = image.img_to_array(img)
            img_array = (img_array / 127.5) - 1.0  # Normalize to [-1, 1]
            images.append(img_array)
    return np.array(images)


# Load the real images from the input folder
real_images = load_images_from_folder(input_folder)

# Create models
generator = build_generator()
discriminator = build_discriminator()
gan = build_gan(generator, discriminator)

# Compile Discriminator
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
discriminator.compile(
    loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"]
)
gan.compile(loss="binary_crossentropy", optimizer=optimizer)

# Training Loop
epochs = 10000
batch_size = 32
half_batch = batch_size // 2

for epoch in range(epochs):
    # Train Discriminator
    idx = np.random.randint(0, real_images.shape[0], half_batch)
    real_imgs = real_images[idx]

    noise = np.random.normal(0, 1, (half_batch, 100))
    fake_imgs = generator.predict(noise)

    d_loss_real = discriminator.train_on_batch(real_imgs, np.ones((half_batch, 1)))
    d_loss_fake = discriminator.train_on_batch(fake_imgs, np.zeros((half_batch, 1)))

    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # Train Generator
    noise = np.random.normal(0, 1, (batch_size, 100))
    g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

    # Print the progress
    if epoch % 1000 == 0:
        print(f"Epoch: {epoch}, D Loss: {d_loss[0]}, G Loss: {g_loss}")

    # Save generated images periodically
    if epoch % 1000 == 0:
        noise = np.random.normal(0, 1, (1, 100))
        gen_image = generator.predict(noise)

        # Save the generated image to the folder
        gen_image = (gen_image[0] + 1) / 2  # Rescale image to [0, 1]
        plt.imshow(gen_image)
        plt.axis("off")
        plt.savefig(f"{save_dir}/generated_image_epoch_{epoch}.png")
        plt.close()
