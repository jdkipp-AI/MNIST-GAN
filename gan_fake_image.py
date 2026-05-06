# RUN THIS IN: gan_env (conda)

# Step 1: import libraries
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, LeakyReLU, BatchNormalization, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np

# Step 2: load and reprocess the data. 
# Load MNIST data
(X_train, _), (_, _) = mnist.load_data()
# Normalize to between -1 and 1
X_train = (X_train.astype(np.float32) - 127.5) / 127.5
X_train = np.expand_dims(X_train, axis=3)

# Step 3: build generator and discriminator
# Generator
def create_generator():
    model = Sequential()
    model.add(Dense(256, input_dim=100))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))

    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))

    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))

    model.add(Dense(784, activation='tanh'))
    model.add(Reshape((28, 28, 1)))
    return model

# Discriminator
def create_discriminator(dropout_rate=0.3):
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28, 1)))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(dropout_rate))

    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(dropout_rate))

    model.add(Dense(1, activation='sigmoid'))
    return model

# Step 4: complie the models
from tensorflow.keras.models import Sequential, Model

# Optimizer tuned for DCGAN stability
disc_opt = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.999)
gan_opt = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.999)

# Create and compile the discriminator
discriminator = create_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=disc_opt, metrics=['accuracy'])

# Create the generator
generator = create_generator()

# Build the combined GAN model
discriminator.trainable = False
gan_input = Input(shape=(100,))
synthetic = generator(gan_input)
gan_output = discriminator(synthetic)
gan = Model(inputs=gan_input, outputs=gan_output)
gan.compile(loss='binary_crossentropy', optimizer=gan_opt)

# Step 5: train the models
def smooth_positive_labels(size):
    return np.random.uniform(low=0.8, high=1.0, size=(size, 1))


def smooth_negative_labels(size):
    return np.random.uniform(low=0.0, high=0.2, size=(size, 1))


def train(epochs=1, batch_size=128, noise_dim=100, instance_noise_std=0.05):
    # Load and preprocess the data
    (X_train, _), (_, _) = mnist.load_data()
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = np.expand_dims(X_train, axis=3)

    batches_per_epoch = X_train.shape[0] // batch_size

    for epoch in range(epochs):
        for batch in range(batches_per_epoch):
            # ---------------------
            #  Train Discriminator
            # ---------------------
            discriminator.trainable = True

            # Sample real images and apply instance noise
            real_images = X_train[batch * batch_size:(batch + 1) * batch_size]
            if instance_noise_std:
                real_images += np.random.normal(0, instance_noise_std, real_images.shape)
                real_images = np.clip(real_images, -1.0, 1.0)

            real_labels = smooth_positive_labels(batch_size)

            d_loss_real, d_acc_real = discriminator.train_on_batch(real_images, real_labels)

            # Generate fake images with new noise sample
            noise = np.random.normal(0, 1, (batch_size, noise_dim))
            fake_images = generator.predict_on_batch(noise)
            if instance_noise_std:
                fake_images += np.random.normal(0, instance_noise_std, fake_images.shape)
                fake_images = np.clip(fake_images, -1.0, 1.0)

            fake_labels = smooth_negative_labels(batch_size)

            d_loss_fake, d_acc_fake = discriminator.train_on_batch(fake_images, fake_labels)

            d_loss = 0.5 * (d_loss_real + d_loss_fake)
            d_acc = 0.5 * (d_acc_real + d_acc_fake)

            # ---------------------
            #  Train Generator
            # ---------------------
            discriminator.trainable = False

            # Noise input for generator (fresh sample)
            noise = np.random.normal(0, 1, (batch_size, noise_dim))
            # Label smoothing with occasional label flipping
            valid_y = smooth_positive_labels(batch_size)

            g_loss = gan.train_on_batch(noise, valid_y)

            if batch % 50 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs} | Batch {batch}/{batches_per_epoch} | "
                    f"D Loss: {d_loss:.4f} (acc {d_acc:.3f}) | G Loss: {g_loss:.4f}"
                )

# Step 6: execute training
# Call the train function with adjusted defaults for stability
train(epochs=10, batch_size=256, noise_dim=100, instance_noise_std=0.05)

generator.save('my_mnist_generator.h5')

#Step 7: genrate new image and evaluate model performance
import matplotlib.pyplot as plt
# Generate random noise as an input to initialize the generator
random_noise = np.random.normal(0,1, [100, 100])

# Generate the images from the noise
generated_images = generator.predict(random_noise)

# Visualize the generated images
plt.figure(figsize=(10,10))
for i in range(generated_images.shape[0]):
    plt.subplot(10, 10, i+1)
    plt.imshow(generated_images[i, :, :, 0], cmap='gray')
    plt.axis('off')
plt.tight_layout()
plt.show()

