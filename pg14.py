import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, SimpleRNN, Dense

# Time-series data
data = np.arange(10, 110, 2, dtype=float)
print("Original data:")
print(data)

# Normalize
data_min = data.min()
data_max = data.max()

scaled = (data - data_min) / (data_max - data_min)

# Create sequences
X, y = [], []

for i in range(len(scaled) - 3):
    X.append(scaled[i:i+3])
    y.append(scaled[i+3])

X = np.array(X)
y = np.array(y)

print("\nInput sequences (X):")
print(X)
print("\nTarget values (y):")
print(y)

# Reshape for RNN
X = X.reshape(X.shape[0], X.shape[1], 1)

# RNN model
model = Sequential([
    Input(shape=(3, 1)),
    SimpleRNN(10, activation='tanh'),
    Dense(1)
])

model.compile(
    optimizer='adam',
    loss='mse'
)

# Train
model.fit(X, y, epochs=500, verbose=0)

# Last 3 values
last_values = np.array([104, 106, 108], dtype=float)

last_scaled = (last_values - data_min) / (data_max - data_min)
last_scaled = last_scaled.reshape(1, 3, 1)

# Predict
prediction_scaled = model.predict(last_scaled, verbose=0)

# Convert back
prediction = (
    prediction_scaled[0][0] * (data_max - data_min)
    + data_min
)

print("Next predicted value:", round(prediction, 2))