import numpy as np

from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler


# Load Iris dataset
iris = load_iris()

X = iris.data


# Scale values between 0 and 1
X = MinMaxScaler().fit_transform(X)


# Parameters
vigilance = 0.7
learning_rate = 0.5


# Start with no clusters
weights = []


# ART training
for x in X:

    # Create first cluster
    if len(weights) == 0:
        weights.append(x.copy())
        continue

    # Calculate similarity
    similarity = []

    for w in weights:
        s = np.sum(np.minimum(x, w)) / np.sum(x)
        similarity.append(s)

    # Find best matching cluster
    winner = np.argmax(similarity)

    # Vigilance test
    if similarity[winner] >= vigilance:

        # Update cluster
        weights[winner] = (
            learning_rate * x
            + (1 - learning_rate) * weights[winner]
        )

    else:

        # Create new cluster
        weights.append(x.copy())


# Display result
print("Number of Clusters:", len(weights))

print("\nFinal Cluster Weights:")

for i, w in enumerate(weights):
    print(
        "Cluster",
        i + 1,
        ":",
        np.round(w, 2)
    )