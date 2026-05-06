# RUN THIS IN: gan_env (conda)

import os
# 1. Fuel Additive: Intel Turbo & Mute Warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import layers, Sequential
import numpy as np

# 2. Build the "Art Critic" (Discriminator)
# It takes a 28x28 image and "crushes" it down to a single number
critic = Sequential([
    layers.Flatten(input_shape=(28, 28)), # Flatten the grid into a line of 784 pixels
    layers.Dense(512, activation='leaky_relu'), # The "Thinking" layer
    layers.Dense(256, activation='leaky_relu'), # Second opinion
    layers.Dense(1, activation='sigmoid')       # The Final Verdict (0 to 1)
])

print("🧐 The Art Critic has been hired!")
critic.summary()

# 3. Create a "Fake" image (The noise we made earlier)
fake_image = np.random.normal(0, 1, (1, 28, 28))

# 4. Ask the Critic to judge it
verdict = critic.predict(fake_image)

print("\n--- THE VERDICT ---")
# Sigmoid outputs a number between 0 and 1. 
# Near 1 = "I think it's Real!" | Near 0 = "Definitely a Fake!"
print(f"Confidence Score: {verdict[0][0]:.4f}")

if verdict[0][0] > 0.5:
    print("😲 The Critic was FOOLED! It thinks this noise is real.")
else:
    print("🚫 The Critic caught the fake! It knows this is just noise.")