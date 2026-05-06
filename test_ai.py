import tensorflow as tf

print("--- GAN Environment Check ---")
print(f"TensorFlow version: {tf.__version__}")

# Check if TensorFlow can 'see' your CPU/GPU
devices = tf.config.list_physical_devices()
print(f"Available devices: {devices}")

if len(devices) > 0:
    print("✅ Setup Successful! You are ready for GANs.")
else:
    print("❌ Something went wrong. TensorFlow cannot find your hardware.")