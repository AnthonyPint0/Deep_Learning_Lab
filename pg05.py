import numpy as np

# Input data (XOR)
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

Y = np.array([
    [0],
    [1],
    [1],
    [0]
])

# Weights and biases
np.random.seed(1)

W1 = np.random.rand(2, 2)
b1 = np.zeros((1, 2))

W2 = np.random.rand(2, 1)
b2 = np.zeros((1, 1))

lr = 0.1

# Training
for i in range(1000):

    # 1. Forward Pass
    H = np.tanh(np.dot(X, W1) + b1)

    O = 1 / (1 + np.exp(
        -(np.dot(H, W2) + b2)
    ))

    # 2. Loss
    loss = np.mean((Y - O) ** 2)

    # 3. Backpropagation
    dO = (O - Y) * O * (1 - O)

    dW2 = np.dot(H.T, dO)
    db2 = np.sum(dO, axis=0, keepdims=True)

    dH = np.dot(dO, W2.T)
    dH = dH * (1 - H ** 2)

    dW1 = np.dot(X.T, dH)
    db1 = np.sum(dH, axis=0, keepdims=True)

    # 4. Weight Update
    W2 -= lr * dW2
    b2 -= lr * db2

    W1 -= lr * dW1
    b1 -= lr * db1

# Final results
print("Final Loss:", loss)

print("Predicted Output:")
print(np.round(O, 2))

print("\nPredicted Classes:")
print((O >= 0.5).astype(int))

print("\nActual Classes:")
print(Y)