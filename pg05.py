import numpy as np

# --------------------------------------------------
# XOR Input Data
# --------------------------------------------------

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=float)

# XOR Output
# 0 XOR 0 = 0
# 0 XOR 1 = 1
# 1 XOR 0 = 1
# 1 XOR 1 = 0

Y = np.array([
    [0],
    [1],
    [1],
    [0]
], dtype=float)

print(f"Input Data (X):\n{X}\n")
print(f"Output Data (Y):\n{Y}\n")


# --------------------------------------------------
# Initialize Weights and Biases
# --------------------------------------------------

np.random.seed(1)

# Input layer: 2 neurons
# Hidden layer: 2 neurons
W1 = np.random.randn(2, 2) * 0.5
b1 = np.zeros((1, 2))
print(f"Initial W1:\n{W1}")
print(f"Initial b1:\n{b1}")

# Hidden layer: 2 neurons
# Output layer: 1 neuron
W2 = np.random.randn(2, 1) * 0.5
b2 = np.zeros((1, 1))
print(f"\nInitial W2:\n{W2}")
print(f"Initial b2:\n{b2}")

learning_rate = 0.1
epochs = 10000


# --------------------------------------------------
# Activation Functions
# --------------------------------------------------

def tanh(x):
    return np.tanh(x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# --------------------------------------------------
# Training
# --------------------------------------------------

for epoch in range(epochs):

    # ==================================================
    # 1. FORWARD PROPAGATION
    # ==================================================

    # Hidden layer
    Z1 = np.dot(X, W1) + b1
    H = tanh(Z1)

    # Output layer
    Z2 = np.dot(H, W2) + b2
    O = sigmoid(Z2)


    # ==================================================
    # 2. CALCULATE LOSS
    # ==================================================

    loss = np.mean((Y - O) ** 2)


    # ==================================================
    # 3. BACKPROPAGATION
    # ==================================================

    # Output layer gradient
    dO = (O - Y) * O * (1 - O)

    # Gradients for output layer weights and bias
    dW2 = np.dot(H.T, dO)
    db2 = np.sum(dO, axis=0, keepdims=True)

    # Hidden layer gradient
    dH = np.dot(dO, W2.T)

    # Derivative of tanh
    dZ1 = dH * (1 - H ** 2)

    # Gradients for hidden layer weights and bias
    dW1 = np.dot(X.T, dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)


    # ==================================================
    # 4. UPDATE WEIGHTS AND BIASES
    # ==================================================

    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1


# --------------------------------------------------
# FINAL FORWARD PASS
# --------------------------------------------------
# Recalculate output using the trained weights.

Z1 = np.dot(X, W1) + b1
H = tanh(Z1)

Z2 = np.dot(H, W2) + b2
O = sigmoid(Z2)


# --------------------------------------------------
# FINAL LOSS
# --------------------------------------------------

final_loss = np.mean((Y - O) ** 2)


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

prediction = (O >= 0.5).astype(int)


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

print("Final Loss:")
print(round(final_loss, 6))

print("\nPredicted Output:")
print(np.round(O, 4))

print("\nPredicted Classes:")
print(prediction)

print("\nActual Classes:")
print(Y.astype(int))

print("\nAccuracy:")
accuracy = np.mean(prediction == Y) * 100
print(f"{accuracy:.2f}%")