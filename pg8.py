import numpy as np

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler


# Load Iris dataset
iris = load_iris()

X = iris.data


# Normalize data
X = StandardScaler().fit_transform(X)


# SOM size
rows = 2
cols = 2


# Initialize weights
weights = np.random.rand(rows, cols, 4)

learning_rate = 0.5


# Training
for epoch in range(100):

    for x in X:

        # Find Best Matching Unit (BMU)
        distance = np.sum((weights - x) ** 2, axis=2)

        bmu = np.unravel_index(
            np.argmin(distance),
            distance.shape
        )

        # Update BMU
        r, c = bmu

        weights[r, c] += learning_rate * (
            x - weights[r, c]
        )


# Display SOM mapping
print("SOM Cluster Mapping:")

for i, x in enumerate(X):

    distance = np.sum((weights - x) ** 2, axis=2)

    bmu = np.unravel_index(
        np.argmin(distance),
        distance.shape
    )

    print("Sample", i + 1, "-> Node", bmu)