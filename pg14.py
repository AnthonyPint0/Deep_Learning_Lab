import numpy as np
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense

# Sample time-series data
data = np.array([10, 12, 14, 16, 18, 20, 22, 24, 26, 28])

# Create input and output 
X = data[:-1]
y = data[1:]

# Reshape input
X = X.reshape(-1, 1) 

# Create model 
model = Sequential([
Dense(10, activation='relu', input_shape=(1,)), Dense(1)
])

# Compile
model.compile(optimizer='adam', loss='mse') 

# Train
model.fit(X, y, epochs=100, verbose=0) 

# Predict next value
last_value = np.array([[28]]) 
prediction = model.predict(last_value)
print("Next predicted value:", prediction[0][0])
