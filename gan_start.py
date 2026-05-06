# RUN THIS IN: gan_env (conda)

import os
# 1. Fuel Additive: Turn on Intel Speed & Mute Dashboard warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt

# 2. Create the "Imagination" (100 random numbers)
# This is what a GAN uses as a seed to create an image
latent_space_seed = np.random.normal(0, 1, (1, 100))

# 3. Create a 28x28 grid of "AI Noise" (Like a fuzzy TV screen)
# This mimics the size of the famous MNIST handwritten digits
noise_image = np.random.normal(0, 1, (28, 28))

print("🎨 AI Imagination Seed Created!")
print(f"Seed Shape: {latent_space_seed.shape}")

# 4. "Develop" the photo so we can see it
plt.imshow(noise_image, cmap='gray')
plt.title("The GAN's First 'Blank Canvas'")
plt.show()
