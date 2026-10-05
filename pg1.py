import nt

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def tanh(x):
    return np.tanh(x)
def relu(x):
    return np.maximum(0, x)
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

# Input values
x = np.array([-2, -1, 0, 1, 2]) 

while True:

    print("\n===== ACTIVATION FUNCTION MENU =====")
    print("1. Sigmoid")
    print("2. Tanh")
    print("3. ReLU") 
    print("4. Softmax")
    print("5. Compare All Outputs")
    print("6. Plot All Graphs")
    print("7. Exit")
    
    choice = int(input("Enter your choice: ")) 
    
    if choice == 1:
        print("Sigmoid Output:")
        print(sigmoid(x))
    elif choice == 2:
        print("Tanh Output:") 
        print(tanh(x))
    elif choice == 3: 
        print("ReLU Output:") 
        print(relu(x))
    elif choice == 4: 
        print("Softmax Output:")
        print(softmax(x))
    elif choice == 5:
        print("\nInput:", x) 
        print("Sigmoid :", sigmoid(x))
        print("Tanh	:", tanh(x))
        print("ReLU	:", relu(x)) 
        print("Softmax :", softmax(x))
    elif choice == 6:
        x_plot = np.linspace(-5, 5, 100)
        plt.plot(x_plot, sigmoid(x_plot), label="Sigmoid") 
        plt.plot(x_plot, tanh(x_plot), label="Tanh") 
        plt.plot(x_plot, relu(x_plot), label="ReLU") 
        plt.plot(x_plot, softmax(x_plot), label="Softmax") 
        plt.xlabel("Input")
        plt.ylabel("Output")
        plt.title("Comparison of Activation Functions") 
        plt.legend()
        plt.grid() 
        plt.show()
    elif choice == 7: 
        print("Program ended.") 
        break
    else:
        print("Invalid choice! Please try again.")

