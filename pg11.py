import numpy as np

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler


# Load Iris dataset
iris = load_iris()

X = iris.data


# Normalize data
X = StandardScaler().fit_transform(X)


# Initialize 3 cluster centers
weights = X[:3].copy()

learning_rate = 0.1


# Training
for epoch in range(20):

    for x in X:

        # Calculate distance
        distance = np.linalg.norm(
            weights - x,
            axis=1
        )

        # Find winning neuron
        winner = np.argmin(distance)

        # Update winner
        weights[winner] = (
            weights[winner]
            + learning_rate * (x - weights[winner])
        )


# Display clusters
print("Final Cluster Centers:")
print(np.round(weights, 2))


print("\nCluster Assignment:")

for i, x in enumerate(X[:10]):

    # Calculate distance
    distance = np.linalg.norm(
        weights - x,
        axis=1
    )

    # Find winning cluster
    winner = np.argmin(distance)

    print(
        "Sample",
        i + 1,
        "-> Cluster",
        winner + 1
    )