import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


# Generator Model
def build_generator():
    model = tf.keras.Sequential(
        [
            layers.Dense(128, activation="relu", input_dim=100),
            layers.BatchNormalization(momentum=0.8),
            layers.Dense(256, activation="relu"),
            layers.BatchNormalization(momentum=0.8),
            layers.Dense(512, activation="relu"),
            layers.BatchNormalization(momentum=0.8),
            layers.Dense(1024, activation="relu"),
            layers.BatchNormalization(momentum=0.8),
            layers.Dense(64 * 64 * 3, activation="tanh"),
            layers.Reshape((64, 64, 3)),
        ]
    )
    return model


# Discriminator Model
def build_discriminator():
    model = tf.keras.Sequential(
        [
            layers.Flatten(input_shape=(64, 64, 3)),
            layers.Dense(1024, activation="relu"),
            layers.Dense(512, activation="relu"),
            layers.Dense(256, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    return model


# GAN Model
def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = tf.keras.Sequential([generator, discriminator])
    return model
