# RUN THIS IN: gan_env (conda)

import os
# 1. Fuel Additive: Intel Turbo & Mute Warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import layers, Sequential
import matplotlib.pyplot as plt
import numpy as np

# 2. Build the "Art Forger's" First Brain Layer
# This takes 100 random numbers and "thinks" them into 784 pixels (28x28)
model = Sequential([
    # Layer 1: The "Inner Thoughts" (More neurons!)
    layers.Dense(512, input_shape=(100,), activation='leaky_relu'),
    
    # Layer 2: The "Output" (Turning thoughts into pixels)
    layers.Dense(28 * 28, activation='tanh'),
    
    # Layer 3: The "Frame" (Folding pixels into a square)
    layers.Reshape((28, 28))
])

model.summary()

print("🧠 Neural Network Built!")

# 3. Create the "Latent Seed" (The 100 random numbers)
seed = np.random.normal(0, 1, (1, 100))

# 4. Turn the key! Pass the seed THROUGH the brain
# This is where your Intel CPU does the heavy lifting
generated_image = model.predict(seed)

print("🚀 The Brain processed the static into a shape!")

# 5. Take a photo of the result
plt.imshow(generated_image[0], cmap='gray')
plt.title("The 'Untrained' Brain's First Attempt")
plt.savefig("untrained_attempt.png") # Auto-save!
plt.show()