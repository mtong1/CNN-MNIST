"""
Based on 'Neural Network from Scratch' by TheIndependentCode.
Source: https://github.com/TheIndependentCode/Neural-Network/blob/master/reshape.py
"""
import numpy as np
from layer import Layer

class Reshape(Layer):
    """
    A class used to reshape the input in the forward pass and reshape the output gradient in the backward pass.
    """

    def __init__(self, in_shape, out_shape):
        """
        Initialize Reshape layer with input and output shapes.

        Args:
        in_shape (tuple): The shape of the input.
        out_shape (tuple): The shape of the output.
        """
        self.in_shape = in_shape
        self.out_shape = out_shape

    def forward(self, input):
        """
        Reshapes the input to the output shape.

        Args:
        input (np.array): The input to be reshaped.

        Returns:
        np.array: The reshaped input.
        """
        return np.reshape(input, self.out_shape)
    
    def backward(self, output_grad, learning_rate): 
        """
        Reshapes the output gradient to the input shape.

        Args:
        output_grad (np.array): The gradient of the output.
        learning_rate (float): The learning rate for the backward pass.

        Returns:
        np.array: The reshaped output gradient.
        """
        return np.reshape(output_grad, self.in_shape)
