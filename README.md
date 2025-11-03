# CSC173 Activity 01 - Neural Network from Scratch

**Date:** October 13, 2025  
**Team:** [Chriscent Louis June M. Pingol]

<details>
<summary><h2>Project Overview</h2></summary>

This project implements a simple neural network for binary classification using breast cancer diagnostic data. The network is built completely from scratch using only Python and NumPy, with no machine learning libraries. The goal is to deepen understanding of neural network fundamentals including forward propagation, loss computation, backpropagation, gradient descent training, and model evaluation.

## Data Preparation

We used the Breast Cancer Wisconsin Diagnostic dataset obtained from these sources:
- [Scikit-learn breast cancer dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)
- [UCI Machine Learning Repository (Breast Cancer Wisconsin Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)  

We selected two features from the dataset for the input layer of the network.

## Network Architecture

- Input layer: 2 neurons (corresponding to selected features)
- Hidden layer: 2 to 4 neurons, activation function: Sigmoid, ReLU, or Tanh
- Output layer: 1 neuron to produce binary classification output

## Implementation Details

- Weight and bias parameters initialized randomly.
- Forward propagation implements layer-wise computations with chosen activation functions.
- Loss computed using Mean Squared Error (MSE).
- Backpropagation calculates gradients of weights and biases.
- Parameters updated using gradient descent.
- Training performed for 500 to 1000 iterations.

## Results & Visualization


## Team Collaboration

Each member contributed to different components of the network:
- Weight and bias initialization
- Forward propagation coding
- Loss function implementation
- Backpropagation and gradient computation
- Training loop and visualization

## How to Run

1. Clone the GitHub repository:
   ```
   git clone https://github.com/KishonShrill/csc173-neural-network-pingol
   ```
2. Open the Jupyter notebook or Colab file.
3. Run all cells sequentially.
4. Explore training loss plot and decision boundary visualizations.

## Summary

This activity provided hands-on experience in building a neural network without relying on high-level ML frameworks. The group collaboratively developed the model, analyzed its training behavior visually, and demonstrated understanding of fundamental AI concepts through both code and documentation.

Video: [link]()

</details>

# CSC173 Activity 02 - Neural Network from Scratch Using PyTorch and Custom Dataset

**Date:** November 3, 2025  
**Team:** [Chriscent Louis June M. Pingol]

<details>
   <summary><h2>Project Overview</h2></summary>

   In this activity, we built a neural network from scratch using PyTorch to predict whether a student will pass or fail based on their study hours and attendance rate.

We started by generating a synthetic dataset where students who study more than 5 hours and attend more than 70% of classes are labeled as “pass,” while those below those thresholds are labeled as “fail.” We explored and visualized the dataset to see clear group separations between passing and failing students.

Next, we prepared the data by normalizing the features and splitting it into training and testing sets. We then implemented a simple feedforward neural network with two hidden layers (8 and 4 neurons) using ReLU activations and a Sigmoid output for binary classification.

We used Binary Cross-Entropy Loss and the Adam optimizer to train the model for 100 epochs. After training, we evaluated its performance using accuracy and visualized the loss curve and decision boundary, showing how well the network learned to distinguish between passing and failing students.

   The project file is uploaded in this [location](https://github.com/KishonShrill/csc173-neural-network-pingol/blob/main/CSC173_Activity02_Pingol_Chriscent_Louis_June.ipynb).
</details>
