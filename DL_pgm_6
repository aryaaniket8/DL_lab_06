USING CONVULATIONAL LSTM

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical

# Parameters
num_notes = 50
sequence_length = 30
num_sequences = 1000

# Generate synthetic data
np.random.seed(42)
data = np.random.randint(0, num_notes, size=(num_sequences, sequence_length, 1))

# One-hot encode data
x_train = to_categorical(data, num_classes=num_notes)

# ConvLSTM Model
inputs = layers.Input(shape=(sequence_length, 1, num_notes))
x = layers.ConvLSTM2D(64, (3, 3), return_sequences=True, activation="relu")(inputs)
x = layers.ConvLSTM2D(64, (3, 3), return_sequences=False, activation="relu")(x)
x = layers.Flatten()(x)
x = layers.Dense(sequence_length * num_notes, activation="softmax")(x)
outputs = layers.Reshape((sequence_length, num_notes))(x)

model = models.Model(inputs, outputs)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(x_train, x_train, epochs=20, batch_size=64, validation_split=0.2)

# Music Generation Function
def gen_mus(model, sequence_length, num_notes, num_gen=50):
    generated = []
    start = np.random.randint(0, num_notes, size=(1, sequence_length, 1, num_notes))
    for _ in range(num_gen):
        pred = model.predict(start, verbose=0)
        next_note = np.argmax(pred[0, -1])
        generated.append(next_note)
        next_onehot = to_categorical(next_note, num_classes=num_notes).reshape(1, 1, 1, num_notes)
        start = np.concatenate([start[:, 1:], next_onehot], axis=1)
    return generated

# Generate and Plot
generated_music = gen_mus(model, sequence_length, num_notes)

def plot_piano_roll(generated_music, num_notes):
    piano_roll = np.zeros((num_notes, len(generated_music)))
    for t, note in enumerate(generated_music):
        piano_roll[note, t] = 1

    plt.figure(figsize=(15, 6))
    sns.heatmap(
        piano_roll,
        cmap="coolwarm",
        cbar=True,
        xticklabels=10,
        yticklabels=True,
        linewidths=0.1,
        linecolor='gray'
    )
    plt.title("Piano Roll Representation of Generated Music")
    plt.xlabel("Time Steps")
    plt.ylabel("Notes")
    plt.yticks(ticks=np.arange(0, num_notes, step=5), labels=np.arange(0, num_notes, step=5))
    plt.xticks(ticks=np.arange(0, len(generated_music), step=5), labels=np.arange(0, len(generated_music), step=5))
    plt.show()

print("Generated music sequence:", generated_music)
plot_piano_roll(generated_music, num_notes)
