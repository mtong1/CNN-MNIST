import numpy as np
from layer import Layer

class Dense(Layer):
    """
    Dense layer class for a neural network. Inherits from the Layer class.
    """

    def __init__(self, input_size, output_size):
        """
        Initialize the dense layer with random weights and bias.

        Parameters:
        input_size (int): The size of the input vector.
        output_size (int): The size of the output vector.
        """
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)

    def forward(self, input):
        """
        Perform the forward pass of the dense layer.

        Parameters:
        input (np.array): The input data.

        Returns:
        np.array: The result of the forward pass.
        """
        self.input = input
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient, learning_rate):
        """
        Perform the backward pass of the dense layer.

        Parameters:
        output_gradient (np.array): The gradient of the output.
        learning_rate (float): The learning rate for the update step.

        Returns:
        np.array: The gradient of the input.
        """
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return input_gradient
