import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# Step 1: Input Data (XOR Gate)
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=float)

# Step 2: Output Data
y = np.array([
    [0],
    [1],
    [1],
    [0]
], dtype=float)

# Step 3: Create Neural Network
model = Sequential([
    Dense(4, input_dim=2, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Step 4: Compile Model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Step 5: Train Model
model.fit(X, y, epochs=500, verbose=0)

# Step 6: Test Model
predictions = model.predict(X)

print("\nPredictions:")
for i in range(len(X)):
    print(f"Input: {X[i]} -> Predicted: {predictions[i][0]:.4f}")

# Step 7: Evaluate Model
loss, accuracy = model.evaluate(X, y, verbose=0)

print("\nLoss:", loss)
print("Accuracy:", accuracy)