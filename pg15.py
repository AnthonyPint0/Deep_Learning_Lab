import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, GRU, Dense

# Data
data = np.array([
    10, 12, 14, 16, 18, 20, 22, 24,
    26, 28, 30, 32, 34, 36, 38, 40
])

# Create sequences
X, y = [], []

for i in range(len(data) - 3):
    X.append(data[i:i+3])
    y.append(data[i+3])

X = np.array(X)
y = np.array(y)

print("Input sequences (X):")
print(X)
print("\nTarget values (y):")
print(y)

# Reshape: samples, timesteps, features
X = X.reshape(X.shape[0], X.shape[1], 1)


# -------- LSTM --------
tf.keras.utils.set_random_seed(42)

lstm = Sequential([
    Input(shape=(3, 1)),
    LSTM(10),
    Dense(1)
])

lstm.compile(optimizer='adam', loss='mse')
lstm.fit(X, y, epochs=300, verbose=0)

lstm_loss = lstm.evaluate(X, y, verbose=0)


# -------- GRU --------
tf.keras.utils.set_random_seed(42)

gru = Sequential([
    Input(shape=(3, 1)),
    GRU(10),
    Dense(1)
])

gru.compile(optimizer='adam', loss='mse')
gru.fit(X, y, epochs=300, verbose=0)

gru_loss = gru.evaluate(X, y, verbose=0)


# -------- Comparison --------
print("LSTM Final Loss:", round(lstm_loss, 4))
print("GRU Final Loss :", round(gru_loss, 4))

if lstm_loss < gru_loss:
    print("LSTM has lower loss.")
else:
    print("GRU has lower loss.")