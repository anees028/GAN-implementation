import tensorflow as tf
from tensorflow.keras import layers, models


def build_discriminator(input_shape=(2560, 1440, 3)):
    model = models.Sequential(
        [
            layers.InputLayer(
                input_shape=input_shape
            ),  # Expected shape: (2560, 1440, 3)
            layers.Flatten(),
            layers.Dense(1024, activation="relu"),
            layers.Dense(512, activation="relu"),
            layers.Dense(256, activation="relu"),
            layers.Dense(1, activation="sigmoid"),  # Output: 1 for real, 0 for fake
        ]
    )
    return model


# This file contains the definition of the Discriminator model. The Discriminator is a neural network that
# classifies images as either real (1) or fake (0).
# It uses a simple neural network structure to perform binary classification.
