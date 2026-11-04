# ==========================================
# Experiment 11 : Neural Style Transfer
# Google Colab Compatible
# ==========================================

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image

# ---------------------------------------
# Load Image
# ---------------------------------------

IMG_SIZE = 224

def load_img(path):
    img = image.load_img(path, target_size=(IMG_SIZE, IMG_SIZE))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return preprocess_input(img)

content_image = load_img("content.jpg")
style_image = load_img("style.jpg")

# ---------------------------------------
# Load VGG19
# ---------------------------------------

vgg = VGG19(weights='imagenet', include_top=False)
vgg.trainable = False

content_layer = "block5_conv2"

style_layers = [
    "block1_conv1",
    "block2_conv1",
    "block3_conv1",
    "block4_conv1",
    "block5_conv1"
]

outputs = [vgg.get_layer(name).output for name in style_layers + [content_layer]]

model = Model(vgg.input, outputs)

# ---------------------------------------
# Gram Matrix
# ---------------------------------------

def gram_matrix(feature):

    feature = tf.reshape(feature, (-1, feature.shape[-1]))

    gram = tf.matmul(feature, feature, transpose_a=True)

    return gram

# ---------------------------------------
# Extract Features
# ---------------------------------------

content_features = model(content_image)
style_features = model(style_image)

print("Content Features Shape:")

for f in content_features:
    print(f.shape)

print("\nStyle Features Shape:")

for f in style_features:
    print(f.shape)

# ---------------------------------------
# Display Images
# ---------------------------------------

plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.imshow(tf.keras.utils.load_img("content.jpg"))
plt.title("Content Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(tf.keras.utils.load_img("style.jpg"))
plt.title("Style Image")
plt.axis("off")

plt.show()

# ---------------------------------------
# Display Gram Matrix
# ---------------------------------------

gram = gram_matrix(style_features[0][0])

plt.figure(figsize=(6,6))
plt.imshow(gram)
plt.title("Gram Matrix")
plt.colorbar()
plt.show()

print("\nNeural Style Transfer pipeline created successfully!")
print("A complete optimization process requires gradient descent over many iterations.")