import tensorflow as tf
import time

print("TensorFlow Version:", tf.__version__)

gpus = tf.config.list_physical_devices('GPU')
print("GPU Devices:", gpus)

if gpus:
    print("GPU is available.")
else:
    print("GPU is NOT available. Running on CPU.")

x = tf.random.normal([10000, 10000])

start = time.time()
y = tf.matmul(x, x)
end = time.time()

print("Computation Time:", end - start, "seconds")