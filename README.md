# 🧠 Deep Learning Lab

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg?logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243.svg?logo=numpy&logoColor=white)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c.svg?logo=python&logoColor=white)](https://matplotlib.org/)

A comprehensive collection of **15 Deep Learning laboratory experiments and practical implementations**. This repository covers foundational neural network math, classical neural architectures, competitive learning paradigms, modern deep learning frameworks, computer vision models, and sequential recurrent architectures.

---

## 📑 Table of Contents

- [🧠 Deep Learning Lab](#-deep-learning-lab)
  - [📑 Table of Contents](#-table-of-contents)
  - [🔬 Overview](#-overview)
  - [📂 Repository Structure \& Index](#-repository-structure--index)
  - [🔍 Detailed Experiment Breakdown](#-detailed-experiment-breakdown)
    - [1. Foundations \& Perceptrons](#1-foundations--perceptrons)
    - [2. Classical \& Competitive Learning Networks](#2-classical--competitive-learning-networks)
    - [3. Deep Neural Networks \& Computer Vision](#3-deep-neural-networks--computer-vision)
    - [4. Sequential \& Recurrent Architectures](#4-sequential--recurrent-architectures)
  - [🛠️ Prerequisites \& Installation](#️-prerequisites--installation)
    - [Requirements](#requirements)
    - [Setup Virtual Environment](#setup-virtual-environment)
    - [Install Dependencies](#install-dependencies)
  - [🚀 How to Run](#-how-to-run)
  - [💻 Tech Stack](#-tech-stack)
  - [📄 License](#-license)

---

## 🔬 Overview

This repository is structured for academic coursework, laboratory practicals, and self-study in Artificial Neural Networks and Deep Learning. It provides:

- **Scratch Implementations:** Step-by-step mathematical coding of gradient descent, perceptrons, backpropagation, RBF, LVQ, SOM, and ART without high-level black-box abstractions.
- **Framework-Powered Models:** Deep neural networks, CNNs, Transfer Learning, and RNNs (LSTM & GRU) implemented using **TensorFlow** and **Keras**.
- **Interactive & Visual Tools:** Menu-driven execution and Matplotlib visualizations for activation responses, decision boundaries, training curves, and confusion matrices.

---

## 📂 Repository Structure & Index

| File                 | Experiment / Topic                          | Key Concepts & Algorithms                                               | Dataset               | Mock Test |
| :------------------- | :------------------------------------------ | :---------------------------------------------------------------------- | :-------------------- | :-------- |
| [`pg01.py`](pg01.py) | **Activation Functions**                    | Sigmoid, Tanh, ReLU, Softmax, formula evaluation & curves               | Synthetic / Array     | ✅        |
| [`pg02.py`](pg02.py) | **Single Layer Perceptron**                 | Perceptron learning rule, linearly separable logic (AND, OR)            | Binary Truth Tables   | ✅        |
| [`pg03.py`](pg03.py) | **Gradient Descent Optimization**           | Univariate Linear Regression, analytical gradients, MSE loss            | Synthetic Linear Data | ❌        |
| [`pg04.py`](pg04.py) | **Multilayer Perceptron (MLP)**             | Full forward & backward propagation, gradient chain rule                | Loan Approval Dataset | ✅        |
| [`pg05.py`](pg05.py) | **XOR Classification with MLP**             | Non-linear classification, hidden layer representation, backpropagation | Non-linear XOR Gate   | ✅        |
| [`pg06.py`](pg06.py) | **Radial Basis Function (RBF) Network**     | Gaussian RBF kernel hidden layer, pseudo-inverse weight solver          | Iris Dataset          | ✅        |
| [`pg07.py`](pg07.py) | **Learning Vector Quantization (LVQ)**      | Supervised competitive learning, prototype vectors, Euclidean metric    | Iris Dataset          | ✅        |
| [`pg08.py`](pg08.py) | **Self-Organizing Maps (SOM)**              | Kohonen map, Best Matching Unit (BMU), 2D grid topological clustering   | Iris Dataset          | ✅        |
| [`pg09.py`](pg09.py) | **Adaptive Resonance Theory (ART)**         | Fuzzy ART, vigilance parameter test, dynamic cluster creation           | Iris Dataset          | ✅        |
| [`pg10.py`](pg10.py) | **Dropout Regularization**                  | Overfitting mitigation, dropout layers, validation curve comparison     | MNIST Digits          | ❌        |
| [`pg11.py`](pg11.py) | **Competitive Learning**                    | Winner-Take-All (WTA) unsupervised clustering                           | Iris Dataset          | ❌        |
| [`pg12.py`](pg12.py) | **Convolutional Neural Networks (CNN)**     | Conv2D, Max Pooling, feature extraction, confusion matrix               | MNIST Digits          | ✅        |
| [`pg13.py`](pg13.py) | **Transfer Learning with VGG16**            | Pretrained feature extractor, spatial adaptation, fine-tuning           | Iris (2D transformed) | ✅        |
| [`pg14.py`](pg14.py) | **Time Series Forecasting (MLP)**           | Sequential autoregression, windowed inputs, next-step prediction        | Numerical Sequence    | ✅        |
| [`pg15.py`](pg15.py) | **Recurrent Neural Networks (LSTM vs GRU)** | Sequence learning, gating mechanisms, loss & performance comparison     | Numerical Sequence    | ✅        |

---

## 🔍 Detailed Experiment Breakdown

### 1. Foundations & Perceptrons

- **[`pg01.py`](pg01.py) - Activation Functions & Plotting**
  - Implements **Sigmoid**, **Hyperbolic Tangent (Tanh)**, **Rectified Linear Unit (ReLU)**, and **Softmax**.
  - Provides an interactive CLI menu to inspect numerical outputs and plot continuous curves using `matplotlib`.

- **[`pg02.py`](pg02.py) - Single Layer Perceptron (SLP)**
  - Implements Rosenblatt's Perceptron learning algorithm from scratch.
  - Demonstrates linear separability by training decision boundaries for **AND** and **OR** logic gates.

- **[`pg03.py`](pg03.py) - Gradient Descent Optimization**
  - Pure Python implementation of Batch Gradient Descent to minimize Mean Squared Error (MSE).
  - Iteratively updates slope ($m$) and intercept ($b$) for linear curve fitting.

- **[`pg04.py`](pg04.py) - Multilayer Perceptron with Backpropagation**
  - Full NumPy implementation of a 2-layer neural network with input, hidden, and output layers.
  - Derives gradients via the chain rule to update weights and biases on a multi-feature loan approval dataset.

- **[`pg05.py`](pg05.py) - Solving the Non-Linear XOR Problem**
  - Overcomes the single-layer perceptron limitation by employing a hidden layer with non-linear activation (Tanh) and output activation (Sigmoid).

---

### 2. Classical & Competitive Learning Networks

- **[`pg06.py`](pg06.py) - Radial Basis Function (RBF) Network**
  - Transforms input space into non-linear Gaussian kernel distances relative to chosen prototype centers.
  - Computes optimal output weights analytically via Moore-Penrose pseudo-inverse (`np.linalg.pinv`).

- **[`pg07.py`](pg07.py) - Learning Vector Quantization (LVQ)**
  - Supervised competitive network with class prototype vectors.
  - Rewards the winner neuron for correct classification and penalizes it for incorrect classification.

- **[`pg08.py`](pg08.py) - Self-Organizing Maps (SOM)**
  - Implements Kohonen's unsupervised clustering network.
  - Maps multi-dimensional Iris features onto a 2D topological grid ($2 \times 2$) by computing Best Matching Units (BMUs).

- **[`pg09.py`](pg09.py) - Adaptive Resonance Theory (ART)**
  - Demonstrates stability-plasticity dilemma resolution using Fuzzy ART principles.
  - Features dynamic cluster allocation regulated by a user-defined vigilance threshold ($\rho = 0.7$).

- **[`pg11.py`](pg11.py) - Competitive Learning (Winner-Take-All)**
  - Unsupervised clustering where the winning neuron closest to the input sample updates its weights towards the input vector.

---

### 3. Deep Neural Networks & Computer Vision

- **[`pg10.py`](pg10.py) - Regularization via Dropout**
  - Trains two Keras models on the MNIST handwritten digits dataset: baseline vs. dropout-regularized (rate: 0.5).
  - Plots side-by-side validation accuracy and loss curves over training epochs to illustrate overfitting prevention.

- **[`pg12.py`](pg12.py) - Convolutional Neural Network (CNN)**
  - End-to-end computer vision pipeline using `Conv2D`, `MaxPooling2D`, and dense classification layers on MNIST.
  - Evaluates performance using a graphical Scikit-Learn confusion matrix.

- **[`pg13.py`](pg13.py) - Transfer Learning with VGG16**
  - Demonstrates deep transfer learning by utilizing a pretrained VGG16 backbone combined with `GlobalAveragePooling2D` and softmax classification.

---

### 4. Sequential & Recurrent Architectures

- **[`pg14.py`](pg14.py) - Time-Series Forecasting with MLP**
  - Formulates continuous sequence forecasting as a supervised learning regression problem using a multi-layer perceptron.

- **[`pg15.py`](pg15.py) - Recurrent Neural Networks: LSTM vs. GRU**
  - Prepares sequential sliding windows from time-series data.
  - Trains and benchmarks **Long Short-Term Memory (LSTM)** against **Gated Recurrent Unit (GRU)** architectures.

---

## 🛠️ Prerequisites & Installation

### Requirements

- **Python 3.8+**
- Git

### Setup Virtual Environment

```bash
# Clone the repository
git clone https://github.com/AnthonyPint0/Deep_Learning_Lab.git
cd Deep_Learning_Lab

# Create and activate a virtual environment
# Windows (PowerShell):
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS:
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash
pip install numpy matplotlib scikit-learn tensorflow
```

---

## 🚀 How to Run

Execute any lab program directly with Python:

```bash
# Activation functions (Interactive Menu + Plotting)
python pg01.py

# Perceptron for logic gates
python pg02.py

# Multilayer Perceptron backpropagation
python pg04.py

# Self-Organizing Maps
python pg08.py

# CNN classification on MNIST digits
python pg12.py

# LSTM vs GRU comparison
python pg15.py
```

---

## 💻 Tech Stack

- **Core Language:** [Python](https://www.python.org/)
- **Numerical Computing:** [NumPy](https://numpy.org/)
- **Deep Learning Framework:** [TensorFlow / Keras](https://www.tensorflow.org/)
- **Machine Learning & Preprocessing:** [scikit-learn](https://scikit-learn.org/)
- **Data Visualization:** [Matplotlib](https://matplotlib.org/)

---

## 📄 License

This repository is maintained for academic and educational purposes. Feel free to use and adapt the code for learning and research.
