# RUN THIS IN: gan_env (conda)

import os
# 1. Fuel Additive: Intel Turbo & Mute Warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import layers, Sequential
import numpy as np

# --- STEP A: THE FORGER (Generator) ---
# Goal: Turn 100 random numbers into a 28x28 image
forger = Sequential([
    layers.Dense(512, input_shape=(100,), activation='leaky_relu'),
    layers.Dense(28 * 28, activation='tanh'),
    layers.Reshape((28, 28))
], name="Forger")

# --- STEP B: THE CRITIC (Discriminator) ---
# Goal: Look at a 28x28 image and say "Real" or "Fake"
critic = Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(256, activation='leaky_relu'),
    layers.Dense(1, activation='sigmoid')
], name="Critic")

print("🏁 The Race is Starting! Both cars are on the track.")

# --- STEP C: THE INTERACTION (The "Thought" Process) ---

# 1. The Forger generates his "Fake Masterpiece" from a random seed
seed = np.random.normal(0, 1, (1, 100))
fake_painting = forger.predict(seed, verbose=0)

# 2. The Critic looks at the fake painting and gives a verdict
verdict = critic.predict(fake_painting, verbose=0)

# --- STEP D: THE BATTLE REPORT ---
print("\n" + "="*30)
print("       GAN BATTLE REPORT       ")
print("="*30)
print(f"Forger's Complexity: {forger.count_params():,} 'Brush Strokes'")
print(f"Critic's Complexity: {critic.count_params():,} 'Eye for Detail'")
print("-" * 30)
print(f"Critic's Verdict (0=Fake, 1=Real): {verdict[0][0]:.4f}")

if verdict[0][0] > 0.5:
    print("🏆 FORGER WINS: The Critic was fooled!")
else:
    print("🚓 CRITIC WINS: The forgery was detected!")
print("="*30)