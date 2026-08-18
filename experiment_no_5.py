import tensorflow as tf
from tensorflow.keras import datasets, layers, models

# Step 1: Load MNIST Dataset
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# Step 2: Normalize the images
train_images = train_images / 255.0
test_images = test_images / 255.0

# Step 3: Reshape images for CNN
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# Step 4: Build CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Step 5: Compile Model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Step 6: Train Model
model.fit(train_images, train_labels, epochs=5)

# Step 7: Evaluate Model
test_loss, test_accuracy = model.evaluate(test_images, test_labels)

print("\nTest Accuracy:", test_accuracy)