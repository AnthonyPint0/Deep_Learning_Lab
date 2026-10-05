import numpy as np

# Loan data
# [Income, Credit Score, Loan Amount]
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
])

# Normalize data
X = X / np.max(X, axis=0)

# Initialize weights
np.random.seed(1)

W1 = np.random.rand(3, 4)
b1 = np.zeros((1, 4))

W2 = np.random.rand(4, 1)
b2 = np.zeros((1, 1))

lr = 0.1

# Training
for epoch in range(1000):

    # -------- Forward Pass --------
    Z1 = np.dot(X, W1) + b1
    H = 1 / (1 + np.exp(-Z1))

    Z2 = np.dot(H, W2) + b2
    O = 1 / (1 + np.exp(-Z2))

    # Loss
    loss = np.mean((Y - O) ** 2)

    # -------- Backward Pass --------
    dO = (O - Y) * O * (1 - O)

    dW2 = np.dot(H.T, dO)
    db2 = np.sum(dO, axis=0, keepdims=True)

    dH = np.dot(dO, W2.T)

    dZ1 = dH * H * (1 - H)

    dW1 = np.dot(X.T, dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    # -------- Weight Update --------
    W2 = W2 - lr * dW2
    b2 = b2 - lr * db2

    W1 = W1 - lr * dW1
    b1 = b1 - lr * db1

# Prediction
prediction = (O >= 0.5).astype(int)

print("Predicted Classes:")
print(prediction.flatten())

print("\nActual Classes:")
print(Y.flatten())

print("\nFinal Loss:", loss)