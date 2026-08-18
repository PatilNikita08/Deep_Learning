import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# Step 1: Sample sequence data
X = np.array([
    [[1], [2], [3]],
    [[2], [3], [4]],
    [[3], [4], [5]],
    [[4], [5], [6]]
], dtype=float)

y = np.array([4, 5, 6, 7], dtype=float)

# Step 2: Build RNN Model
model = Sequential([
    SimpleRNN(10, activation='tanh', input_shape=(3, 1)),
    Dense(1)
])

# Step 3: Compile Model
model.compile(optimizer='adam', loss='mse')

# Step 4: Train Model
model.fit(X, y, epochs=300, verbose=0)

# Step 5: Test the Model
test = np.array([[[5], [6], [7]]], dtype=float)
prediction = model.predict(test, verbose=0)

print("Input Sequence: [5, 6, 7]")
print("Predicted Next Value:", prediction[0][0])