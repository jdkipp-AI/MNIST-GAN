# RUN THIS IN: gan_env (conda)
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, LeakyReLU, BatchNormalization, Dropout
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

# --- Step 1: Functions to build the models ---
def create_generator():
    model = Sequential([
        Dense(256, input_dim=100),
        LeakyReLU(alpha=0.2),
        BatchNormalization(momentum=0.8),
        Dense(512),
        LeakyReLU(alpha=0.2),
        BatchNormalization(momentum=0.8),
        Dense(1024),
        LeakyReLU(alpha=0.2),
        BatchNormalization(momentum=0.8),
        Dense(784, activation='tanh'),
        Reshape((28, 28, 1))
    ])
    return model

def create_discriminator(dropout_rate=0.3):
    model = Sequential([
        Flatten(input_shape=(28, 28, 1)),
        Dense(512),
        LeakyReLU(alpha=0.2),
        Dropout(dropout_rate),
        Dense(256),
        LeakyReLU(alpha=0.2),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
    ])
    return model

# --- Step 2: Training Helper Functions ---
def smooth_positive_labels(size):
    return np.random.uniform(low=0.8, high=1.0, size=(size, 1))

def smooth_negative_labels(size):
    return np.random.uniform(low=0.0, high=0.2, size=(size, 1))

def train(epochs=1, batch_size=128, noise_dim=100, instance_noise_std=0.01):
    # Load and preprocess
    (X_train, _), (_, _) = mnist.load_data()
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = np.expand_dims(X_train, axis=3)
    batches_per_epoch = X_train.shape[0] // batch_size

    for epoch in range(epochs):
        for batch in range(batches_per_epoch):
            # Train Discriminator
            discriminator.trainable = True
            real_images = X_train[batch * batch_size:(batch + 1) * batch_size]
            if instance_noise_std:
                real_images += np.random.normal(0, instance_noise_std, real_images.shape)
            
            d_loss_real, _ = discriminator.train_on_batch(real_images, smooth_positive_labels(batch_size))

            noise = np.random.normal(0, 1, (batch_size, noise_dim))
            fake_images = generator.predict_on_batch(noise)
            d_loss_fake, _ = discriminator.train_on_batch(fake_images, smooth_negative_labels(batch_size))

            # Train Generator
            discriminator.trainable = False
            noise = np.random.normal(0, 1, (batch_size, noise_dim))
            g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1))) # Aim for 'Real'

        print(f"Epoch {epoch+1}/{epochs} | D Loss: {0.5*(d_loss_real+d_loss_fake):.4f} | G Loss: {g_loss:.4f}")

# --- Step 3: Initialize or Load ---
model_path = 'my_mnist_generator.h5'
disc_opt = Adam(learning_rate=0.0002, beta_1=0.5)
gan_opt = Adam(learning_rate=0.0002, beta_1=0.5)

if os.path.exists(model_path):
    print("Found saved model! Loading it to continue...")
    generator = load_model(model_path)
else:
    print("Building a new generator...")
    generator = create_generator()

discriminator = create_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=disc_opt, metrics=['accuracy'])

# Build Combined GAN
discriminator.trainable = False
gan_input = Input(shape=(100,))
synthetic = generator(gan_input)
gan_output = discriminator(synthetic)
gan = Model(inputs=gan_input, outputs=gan_output)
gan.compile(loss='binary_crossentropy', optimizer=gan_opt)

# --- Step 4: Execute ---
SHOULD_TRAIN = True # Set to False to just generate pictures

if SHOULD_TRAIN:
    train(epochs=50, batch_size=256)
    generator.save(model_path)

# --- Step 5: Final Plot ---
random_noise = np.random.normal(0, 1, [100, 100])
generated_images = generator.predict(random_noise)

plt.figure(figsize=(10, 10))
for i in range(100):
    plt.subplot(10, 10, i+1)
    plt.imshow(generated_images[i, :, :, 0], cmap='gray')
    plt.axis('off')
plt.show()