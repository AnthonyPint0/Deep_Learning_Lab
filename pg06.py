import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# Load Iris dataset
iris = load_iris()

X = iris.data
y = iris.target


# Scale data
scaler = StandardScaler()
X = scaler.fit_transform(X)


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Choose centers
centers = X_train[:3]
sigma = 1.0


# RBF function
def rbf(x, center):
    return np.exp(
        -np.sum((x - center) ** 2) / (2 * sigma ** 2)
    )


# Create RBF features
def rbf_features(X):
    R = np.zeros((len(X), len(centers)))

    for i in range(len(X)):
        for j in range(len(centers)):
            R[i, j] = rbf(X[i], centers[j])

    return R


# Transform data
R_train = rbf_features(X_train)
R_test = rbf_features(X_test)


# Train output weights
W = np.linalg.pinv(R_train).dot(y_train)


# Predict
pred = np.round(R_test.dot(W)).astype(int)


# Accuracy
accuracy = accuracy_score(y_test, pred)


# Display results
print("Predicted Classes:")
print(pred)

print("\nActual Classes:")
print(y_test)

print("\nAccuracy:", accuracy)