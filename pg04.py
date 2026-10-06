import numpy as np
from sklearn.metrics import accuracy_score

# --------------------------------------------------
# Loan Approval Dataset
# [Income, Credit Score, Loan Amount]
# --------------------------------------------------

X = np.array([
    [50, 700, 20],
    [60, 750, 25],
    [30, 600, 30],
    [40, 650, 35],
    [80, 800, 20],
    [25, 550, 40]
], dtype=float)

# 1 = Approved, 0 = Rejected
Y = np.array([
    [1],
    [1],
    [0],
    [0],
    [1],
    [0]
], dtype=float)

print(f"Input Data (X):\n{X}\n")


# --------------------------------------------------
# Normalize Input Data
# --------------------------------------------------

X = X / np.max(X, axis=0)
print(f"Normalized Input Data:\n{X}\n")


# --------------------------------------------------
# Initialize Weights and Biases
# --------------------------------------------------

np.random.seed(1)

# Input layer: 3 neurons
# Hidden layer: 4 neurons
W1 = np.random.rand(3, 4) # 3x4 matrix
b1 = np.zeros((1, 4)) # 1x4 vector
print(f"Initial W1:\n{W1}")
print(f"Initial b1:\n{b1}")

# Hidden layer: 4 neurons
# Output layer: 1 neuron
W2 = np.random.rand(4, 1)
b2 = np.zeros((1, 1))
print(f"\nInitial W2:\n{W2}")
print(f"Initial b2:\n{b2}")

learning_rate = 0.1
epochs = 1000


# --------------------------------------------------
# Sigmoid Activation Function
# --------------------------------------------------

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# --------------------------------------------------
# Training
# --------------------------------------------------

for epoch in range(epochs):

    # ---------------- Forward Propagation ----------------

    # Hidden layer
    Z1 = np.dot(X, W1) + b1
    H = sigmoid(Z1)

    # Output layer
    Z2 = np.dot(H, W2) + b2
    O = sigmoid(Z2)

    # Mean Squared Error Loss
    loss = np.mean((Y - O) ** 2)


    # ---------------- Backward Propagation ----------------

    # Output layer gradient
    dO = (O - Y) * O * (1 - O)

    # Gradients for W2 and b2
    dW2 = np.dot(H.T, dO)
    db2 = np.sum(dO, axis=0, keepdims=True)

    # Hidden layer gradient
    dH = np.dot(dO, W2.T)

    dZ1 = dH * H * (1 - H)

    # Gradients for W1 and b1
    dW1 = np.dot(X.T, dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)


    # ---------------- Update Weights ----------------

    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1


# --------------------------------------------------
# Final Forward Pass
# --------------------------------------------------
# Important: perform a new forward pass after training
# so predictions use the updated weights.

Z1 = np.dot(X, W1) + b1
H = sigmoid(Z1)

Z2 = np.dot(H, W2) + b2
O = sigmoid(Z2)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

prediction = (O >= 0.5).astype(int) # Convert probabilities to binary predictions


# --------------------------------------------------
# Display Results
# --------------------------------------------------

final_loss = np.mean((Y - O) ** 2)

print("Predicted Classes:")
print(prediction.flatten())

print("\nActual Classes:")
print(Y.astype(int).flatten())

print("\nOutput Probabilities:")
print(np.round(O.flatten(), 4))

print("\nFinal Loss:")
print(round(final_loss, 6))

print("\nFinal Weights and Biases:")
print(f"W1:\n{W1}")
print(f"b1:\n{b1}")
print(f"\nW2:\n{W2}")
print(f"b2:\n{b2}")

print("\nAccuracy:")
print(accuracy_score(Y.astype(int).flatten(), prediction.flatten()))