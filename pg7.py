import numpy as np

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Load Iris dataset
iris = load_iris()

X = iris.data
y = iris.target


# Scale data
X = StandardScaler().fit_transform(X)


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Initial prototypes: one for each class
W = np.array([
    X_train[y_train == 0][0],
    X_train[y_train == 1][0],
    X_train[y_train == 2][0]
])

labels = np.array([0, 1, 2])

lr = 0.1


# Training
for epoch in range(20):

    for i in range(len(X_train)):

        # Calculate distance
        distance = np.linalg.norm(
            W - X_train[i],
            axis=1
        )

        # Find winner
        winner = np.argmin(distance)

        # Update prototype
        if labels[winner] == y_train[i]:
            W[winner] += lr * (
                X_train[i] - W[winner]
            )
        else:
            W[winner] -= lr * (
                X_train[i] - W[winner]
            )


# Prediction
pred = []

for x in X_test:

    # Calculate distance
    distance = np.linalg.norm(
        W - x,
        axis=1
    )

    # Find winner
    winner = np.argmin(distance)

    # Assign class label
    pred.append(labels[winner])


# Accuracy
accuracy = accuracy_score(y_test, pred)


# Display results
print("Predicted Classes:")
print(pred)

print("\nActual Classes:")
print(y_test)

print("\nAccuracy:", accuracy)