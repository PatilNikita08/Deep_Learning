# ==========================================
# Experiment 12 : Image Generation using VAE
# Google Colab Compatible
# ==========================================

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.layers import Reshape, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import backend as K

# -----------------------------------
# Load MNIST Dataset
# -----------------------------------

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype("float32") / 255.
x_test = x_test.astype("float32") / 255.

x_train = np.reshape(x_train, (-1, 784))
x_test = np.reshape(x_test, (-1, 784))

print("Training Shape :", x_train.shape)

# -----------------------------------
# Parameters
# -----------------------------------

input_dim = 784
hidden_dim = 256
latent_dim = 2

# -----------------------------------
# Encoder
# -----------------------------------

inputs = Input(shape=(input_dim,))

h = Dense(hidden_dim, activation="relu")(inputs)

z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

# -----------------------------------
# Sampling Layer
# -----------------------------------

def sampling(args):

    z_mean, z_log_var = args

    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim))

    return z_mean + K.exp(0.5 * z_log_var) * epsilon

z = Lambda(sampling)([z_mean, z_log_var])

encoder = Model(inputs, [z_mean, z_log_var, z])

encoder.summary()

# -----------------------------------
# Decoder
# -----------------------------------

latent_inputs = Input(shape=(latent_dim,))

x = Dense(hidden_dim, activation="relu")(latent_inputs)

outputs = Dense(input_dim, activation="sigmoid")(x)

decoder = Model(latent_inputs, outputs)

decoder.summary()

# -----------------------------------
# Complete VAE
# -----------------------------------

outputs = decoder(encoder(inputs)[2])

vae = Model(inputs, outputs)

# -----------------------------------
# Loss Function
# -----------------------------------

reconstruction_loss = binary_crossentropy(inputs, outputs)

reconstruction_loss *= input_dim

kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)

kl_loss = -0.5 * K.sum(kl_loss, axis=-1)

vae_loss = K.mean(reconstruction_loss + kl_loss)

vae.add_loss(vae_loss)

vae.compile(optimizer="adam")

vae.summary()

# -----------------------------------
# Train
# -----------------------------------

vae.fit(
    x_train,
    epochs=10,
    batch_size=128,
    validation_data=(x_test, None)
)

# -----------------------------------
# Generate Images
# -----------------------------------

n = 10

random_points = np.random.normal(size=(n, latent_dim))

generated = decoder.predict(random_points)

generated = generated.reshape(n,28,28)

plt.figure(figsize=(15,3))

for i in range(n):

    plt.subplot(1,n,i+1)

    plt.imshow(generated[i], cmap="gray")

    plt.axis("off")

plt.show()