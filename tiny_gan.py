import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

# 1. THE DATA: Using tiny random "images" for a speed test
print("--- Initializing Tiny GAN (CPU Optimized) ---")
latent_dim = 100

# 2. THE GENERATOR: The "Forger"
generator = tf.keras.Sequential([
    layers.Dense(128, activation="relu", input_shape=(latent_dim,)),
    layers.Dense(784, activation="sigmoid"),
    layers.Reshape((28, 28))
], name="Generator")

# 3. THE DISCRIMINATOR: The "Detective"
discriminator = tf.keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation="relu"),
    layers.Dense(1, activation="sigmoid")
], name="Discriminator")

# 4. COMPILING THE BRAIN
discriminator.compile(optimizer="adam", loss="binary_crossentropy")
discriminator.trainable = False

gan_input = layers.Input(shape=(latent_dim,))
fake_image = generator(gan_input)
gan_output = discriminator(fake_image)
gan = tf.keras.Model(gan_input, gan_output)
gan.compile(optimizer="adam", loss="binary_crossentropy")

# 5. TRAINING LOOP (The "Fight")
print("\nStarting Training... (Press Ctrl+C to stop anytime)")
for step in range(5):
    # Create fake "noise" to start
    noise = np.random.normal(size=(32, latent_dim))
    
    # Train the "Detective" and "Forger"
    g_loss = gan.train_on_batch(noise, np.ones((32, 1)))
    
    print(f"Step {step+1}: AI is learning... Loss: {g_loss:.4f}")

print("\n✅ Training Step Complete! Your CPU handled the math perfectly.")