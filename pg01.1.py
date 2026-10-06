from csv import Error

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1 + np.exp(-x))
def tanh(x):
    return np.tanh(x)
def relu(x):
    return np.maximum(0 , x)
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x/np.sum(exp_x)

def display_plot(data, output, title: str):
    plt.plot(data, output, label = title)
    plt.xlabel("Input")
    plt.ylabel("Output")
    title = f"{title.capitalize()} Chart"
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

x = np.array([-2, -1, 0, 1, 2])

while True:
    print("Activation Function: ")
    print("1. Sigmoid")
    print("2. Tanh")
    print("3. ReLU")
    print("4. Softmax")
    print("5. Comparsion of all Activation Function")
    print("6. Exit")
    try:
        choice = int(input("Enter the choice from the above options: "))
    
        if choice == 1:
            print(sigmoid(x))
            display_plot(x, sigmoid(x), "sigmoid")
        elif choice == 2:
            print(tanh(x))
            display_plot(x, tanh(x), "tanh")
        elif choice == 3:
            print(relu(x))
            display_plot(x, relu(x), "ReLU")
        elif choice == 4:
            print(softmax(x))
            display_plot(x, softmax(x), "softmax")
        elif choice == 5:
            x_plot = np.linspace(-5,5,100)
            plt.plot(x_plot, sigmoid(x_plot), label = "Sigmoid")
            plt.plot(x_plot, tanh(x_plot), label = "Tanh")
            plt.plot(x_plot, relu(x_plot), label = "ReLU")
            plt.plot(x_plot, softmax(x_plot), label = "Softmax")
            plt.xlabel("Input")
            plt.ylabel("Output")
            plt.title("Comparsion of all Activation Function")
            plt.legend()
            plt.grid()
            plt.show()
        elif choice == 6:
            break
        else:
            print("Invalid choice! Please try again.")
    
    except ValueError:
            print("Something went wrong!")