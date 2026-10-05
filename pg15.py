import numpy as np 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense


# Sample data
data = np.array([10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40])

# Create sequences 
X = []
y = []

for i in range(len(data) - 3): 
    X.append(data[i:i+3]) 
    y.append(data[i+3])

X = np.array(X) 
y = np.array(y)

# Reshape for RNN
X = X.reshape((X.shape[0], X.shape[1], 1))


# -------- LSTM Model --------
lstm = Sequential([
LSTM(10, input_shape=(3, 1)), Dense(1)
])


lstm.compile(optimizer='adam', loss='mse')


lstm_history = lstm.fit(X, y, epochs=50, verbose=0)


lstm_loss = lstm.evaluate(X, y, verbose=0)


# -------- GRU Model --------
gru = Sequential([
GRU(10, input_shape=(3, 1)), Dense(1)
])
gru.compile(optimizer='adam', loss='mse')


gru_history = gru.fit(X, y, epochs=50, verbose=0)


gru_loss = gru.evaluate(X, y, verbose=0) 

# Compare
print("LSTM Loss:", lstm_loss) 
print("GRU Loss:", gru_loss)

if lstm_loss < gru_loss: 
    print("LSTM has lower loss.")
else:
    print("GRU has lower loss.")
