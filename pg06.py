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
    X, y, test_size=0.2, random_state=42
)

print(f"Y_train:\n{y_train}\n")

# 1. FIX: One-hot encode the labels for multiclass classification
num_classes = len(np.unique(y))  # Get the number of unique classes. Result = 3
print(f"Number of classes: {num_classes}")
y_train_encoded = np.eye(num_classes)[y_train]  # Converts e.g., 2 into [0, 0, 1]
print(f"One-hot encoded training labels:\n{y_train_encoded}\n")

# 2. FIX: Pick more centers randomly (e.g., 10 centers) to capture dataset diversity
num_centers = 10
np.random.seed(42)
random_indices = np.random.choice(X_train.shape[0], num_centers, replace=False)
centers = X_train[random_indices]
sigma = 1.0

# Vectorized RBF function for much faster execution
def rbf_features(X, centers, sigma):
    # Computes pairwise squared Euclidean distances between X and centers
    dists = np.sum((X[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=-1)
    return np.exp(-dists / (2 * sigma ** 2))

# Transform data
R_train = rbf_features(X_train, centers, sigma)
R_test = rbf_features(X_test, centers, sigma)

# 3. FIX: Train output weights matrix (now shape: [num_centers, num_classes])
W = np.linalg.pinv(R_train).dot(y_train_encoded)

# 4. FIX: Predict by picking the class with the highest output score (argmax)
raw_predictions = R_test.dot(W)
pred = np.argmax(raw_predictions, axis=1)

# Accuracy
accuracy = accuracy_score(y_test, pred)

# Display results
print("Predicted Classes:")
print(pred)
print("\nActual Classes:")
print(y_test)
print("\nAccuracy:", accuracy)
