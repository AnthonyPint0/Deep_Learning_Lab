# Gradient Descent for Linear Regression

# Data
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# Initial values
m = 0       # slope
b = 0       # intercept

lr = 0.01   # learning rate
epochs = 1000
n = len(x)

# Gradient Descent
for i in range(epochs):

    # Predictions
    y_pred = [m * xi + b for xi in x]

    # Calculate gradients
    dm = (-2 / n) * sum(
        x[j] * (y[j] - y_pred[j])
        for j in range(n)
    )

    db = (-2 / n) * sum(
        y[j] - y_pred[j]
        for j in range(n)
    )

    # Update m and b
    m = m - lr * dm
    b = b - lr * db

# Final results
print("Slope (m):", m)
print("Intercept (b):", b)

print("\nPredictions:")

for xi in x:
    print("x =", xi, "y =", m * xi + b)