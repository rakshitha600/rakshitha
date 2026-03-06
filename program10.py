import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# Generate synthetic sequential data
def generate_time_series(batch_size, n_steps):
    freq1, freq2, offsets1, offsets2 = np.random.rand(4, batch_size, 1)
    time = np.linspace(0, 1, n_steps)
    series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10))  # Wave 1
    series += 0.5 * np.sin((time - offsets2) * (freq2 * 20 + 20))  # Wave 2
    series += 0.1 * (np.random.rand(batch_size, n_steps) - 0.5)  # Noise
    return series[..., np.newaxis]

# Prepare dataset
n_steps = 50
batch_size = 1000
X_train = generate_time_series(batch_size, n_steps)
y_train = X_train[:, -1]

X_valid = generate_time_series(200, n_steps)
y_valid = X_valid[:, -1]

# Build RNN model
model = Sequential([
    SimpleRNN(20, return_sequences=False, input_shape=[n_steps, 1]),
    Dense(1)
])

# Compile and train the model
model.compile(loss="mse", optimizer="adam")
model.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid))

# Predict with the model
X_test = generate_time_series(1, n_steps)
y_pred = model.predict(X_test)
print("Predicted value:", y_pred)
