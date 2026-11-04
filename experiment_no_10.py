# ============================================
# Experiment 10: Text Generation using LSTM
# Google Colab Compatible
# ============================================

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
import urllib.request
import random

# --------------------------------------------
# Download Shakespeare dataset
# --------------------------------------------
print("Downloading dataset...")

try:
    url = "https://www.gutenberg.org/cache/epub/100/pg100.txt"
    urllib.request.urlretrieve(url, "shakespeare.txt")

    with open("shakespeare.txt", "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

except:
    text = """
    To be or not to be that is the question.
    Whether tis nobler in the mind to suffer
    The slings and arrows of outrageous fortune.
    """ * 100

# Reduce size for faster training
text = text[:100000]

print("Total Characters:", len(text))

# --------------------------------------------
# Create Character Vocabulary
# --------------------------------------------
chars = sorted(list(set(text)))

char_to_index = {c:i for i,c in enumerate(chars)}
index_to_char = {i:c for i,c in enumerate(chars)}

vocab_size = len(chars)

print("Vocabulary Size:", vocab_size)

# --------------------------------------------
# Create Sequences
# --------------------------------------------
sequence_length = 40

X = []
y = []

for i in range(len(text)-sequence_length):
    seq = text[i:i+sequence_length]
    target = text[i+sequence_length]

    X.append([char_to_index[ch] for ch in seq])
    y.append(char_to_index[target])

X = np.array(X)
y = np.array(y)

print("Total Sequences:", len(X))

# --------------------------------------------
# Normalize
# --------------------------------------------
X = X / float(vocab_size)

X = X.reshape((X.shape[0], sequence_length, 1))

y = to_categorical(y, num_classes=vocab_size)

# --------------------------------------------
# Build LSTM Model
# --------------------------------------------
model = Sequential()

model.add(LSTM(128, input_shape=(sequence_length,1)))
model.add(Dense(vocab_size, activation="softmax"))

model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

model.summary()

# --------------------------------------------
# Train Model
# --------------------------------------------
history = model.fit(
    X,
    y,
    epochs=10,
    batch_size=128
)

# --------------------------------------------
# Text Generation
# --------------------------------------------
start = random.randint(0, len(X)-1)

pattern = X[start]

generated = ""

print("\nGenerated Text:\n")

for i in range(500):

    prediction = model.predict(
        pattern.reshape(1,sequence_length,1),
        verbose=0
    )

    index = np.argmax(prediction)

    result = index_to_char[index]

    generated += result

    pattern = np.append(pattern[1:], index/vocab_size)

print(generated)