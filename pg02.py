import numpy as np


def perceptron_train(X, y, epochs=10, lr=0.1):
    weights = np.zeros(X.shape[1])
    bias = 0

    for _ in range(epochs):
        for i in range(X.shape[0]):
            y_pred = 1 if (np.dot(X[i], weights) + bias) > 0 else 0

            weights += lr * (y[i] - y_pred) * X[i]
            bias += lr * (y[i] - y_pred)

    return weights, bias


def main():
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

    y_and = np.array([0, 0, 0, 1])
    y_or = np.array([0, 1, 1, 1])

    while True:
        print("===== SINGLE LAYER PERCEPTRON =====")
        print("1. AND Gate")
        print("2. OR Gate")
        print("3. Exit")

        choice = input("Enter your choice: ")

        if choice == '1':
            w_and, b_and = perceptron_train(X, y_and)

            print("\nAND Gate Results")
            print("----------------")
            print("Input Output")

            for i in range(len(X)):
                print(f"{X[i]}   {y_and[i]}")

            print(f"\nWeights: {w_and}")
            print(f"Bias: {b_and}\n")

        elif choice == '2':
            w_or, b_or = perceptron_train(X, y_or)

            print("\nOR Gate Results")
            print("----------------")
            print("Input Output")

            for i in range(len(X)):
                print(f"{X[i]}   {y_or[i]}")

            print(f"\nWeights: {w_or}")
            print(f"Bias: {b_or}\n")

        elif choice == '3':
            break

        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()