import numpy as np

# Step 1: Input data (AND Gate)
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Step 2: Expected output
y = np.array([0, 0, 0, 1])

# Step 3: Initialize weights and bias
weights = np.zeros(2)
bias = 0
learning_rate = 0.1
epochs = 10

# Step 4: Train the perceptron
for epoch in range(epochs):
    for i in range(len(X)):
        linear_output = np.dot(X[i], weights) + bias
        prediction = 1 if linear_output >= 0 else 0

        error = y[i] - prediction

        weights += learning_rate * error * X[i]
        bias += learning_rate * error

# Step 5: Display final weights and bias
print("Final Weights:", weights)
print("Final Bias:", bias)

# Step 6: Test the perceptron
print("\nPredictions:")
for i in range(len(X)):
    linear_output = np.dot(X[i], weights) + bias
    prediction = 1 if linear_output >= 0 else 0
    print(f"Input: {X[i]} -> Output: {prediction}")