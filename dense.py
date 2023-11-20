"""
Based on 'Neural Network from Scratch' by TheIndependentCode.
Source: https://github.com/TheIndependentCode/Neural-Network/blob/master/dense.py
"""
import numpy as np
from layer import Layer

class Dense(Layer):
    """
    Dense layer class for a neural network. Inherits from the Layer class.
    """

    def __init__(self, input_size, output_size):
        """
        Initialize the dense layer with random weights and bias.

        Args:
        input_size (int): The size of the input vector.
        output_size (int): The size of the output vector.
        """
        # Initialize weights with random values
        self.weights = np.random.randn(output_size, input_size)
        # Initialize bias with random values
        self.bias = np.random.randn(output_size, 1)

    def forward(self, input):
        """
        Perform the forward pass of the dense layer.

        Args:
        input (np.array): The input data.

        Returns:
        np.array: The result of the forward pass.
        """
        # Store the input for use in the backward pass
        self.input = input
        # Compute the output of the dense layer
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient, learning_rate):
        """
        Perform the backward pass of the dense layer.

        Args:
        output_gradient (np.array): The gradient of the output.
        learning_rate (float): The learning rate for the update step.

        Returns:
        np.array: The gradient of the input.
        """
        # Compute the gradient of the weights
        weights_gradient = np.dot(output_gradient, self.input.T)
        # Compute the gradient of the input
        input_gradient = np.dot(self.weights.T, output_gradient)
        # Update the weights using the computed gradient and the learning rate
        self.weights -= learning_rate * weights_gradient
        # Update the bias using the computed gradient and the learning rate
        self.bias -= learning_rate * output_gradient
        # Return the gradient of the input for use in the previous layer
        return input_gradient
