class Sigmoid(Layer):
    """
    This class represents the Sigmoid activation function for a neural network.
    """
    def __init__(self):
        """
        Initialize the Sigmoid class with the sigmoid function and its derivative.
        """
        pass
    def forward(self, input):
        """
        The sigmoid function.
        
        Parameters:
        x (float): The input to the sigmoid function.

        Returns:
        float: The output of the sigmoid function.
        """
        self.input = input
        return 1 / (1 + np.exp(-self.input))

    def backward(self, output_gradient, learning_rate):
        """
        The derivative of the sigmoid function.
        
        Parameters:
        x (float): The input to the derivative of the sigmoid function.

        Returns:
        float: The output of the derivative of the sigmoid function.
        """
        s = sigmoid(x)
        s_prime =  s * (1 - s)
        return s_prime * output_gradient
