import numpy as np 
from layer import Layer 
import correlate

class Convolution(Layer):
    """
    Class that defines the functions of a Convolutional Neural Network, including the forward propagation. 
    """
    def __init__(self, in_shape, kernel_size, depth):
        """
        Constructor for the Convolutional class.
        Args:
            kernel_size(tuple): size of the kernel, determined by user 
            in_shape(tuple): shape of the input, given by a tuple that defines the input depth, height, and width 
            depth(int): how many input matrices there are, also determines the number of kernels per layer 
        """
        in_depth, in_height, in_width = in_shape # input shape is a tuple that defines each of these parameters in this order 
        self.depth = depth
        self.in_shape = in_shape 
        self.in_depth = in_depth 
        self.output_shape = (depth, in_height - kernel_size + 1, in_width - kernel_size + 1)
        self.kernels_shape = (depth, in_depth, kernel_size, kernel_size)
        # last two are size of matrices in each kernel

        # instantiating randomized kernel values to start
        self.kernels = np.random.randn(*self.kernels_shape)

        # instantiating randomized bias values to start (bias will be the shape of the output)
        self.biases = np.random.randn(*self.output_shape)

    def forward(self, input):
        """
        This function performs the forward propagation of our convolutional neural network. It does the math of correlating the inputs
        with the kernels, then adding the biases in (which is performed beforehand). To understand the math, please reference the README
        and the correlate file. 

        Args: 
            input: The training data represented as a matrix.

        Returns:
            output (np.ndarray): The predicted output represented as a matrix. 
        """
        # when function called and input is given, save the input for the conv. class to remember
        self.input = input

        # copies the biases to output 
        self.output = np.copy(self.biases)

        # traverses the depth of the input and determines output value by correlating the input w each of the kernels 
        for i in range(self.depth):
            for j in range(self.in_depth):
                self.output[i] += correlate.correlate2d(self.input[j],self.kernels[i,j], "valid")

        return self.output

    def backward(self, output_grad, learning_rate):
        '''
        This function performs the backward propagation of our convolutional neural network. It establishes and updates the gradients
        for input and kernels based on the output gradients and each other. Note that this function was not written by our team, but
        it still uses our self-written correlate and convolve functions for the vital math aspects. 
        Args:
            output_grad (np.ndarray): The output gradient, in the form of a matrix.
            learning_rate (int): The learning rate that we want the propagation to run at. 

        Returns:
            input_grad (np.ndarray): The input gradient, in the form of a matrix. 
        '''
        # given the shape of the matrix, establish a similar sized empty gradient matrix 
        kernels_grad = np.zeros(self.kernels_shape) 
        input_grad = np.zeros(self.in_shape)        

        # using two for loops to compute the kernel gradient for each kernel of given depth
        # bias gradient doesn"t need to be calculated since it is the same as the output gradient 
        for i in range(self.depth):
            for j in range(self.in_depth):
                kernels_grad[i,j] = correlate.correlate2d(self.input[j], output_grad[i], "valid")

                # computing input gradient by convolving 
                input_grad[j] += correlate.convolve2d(output_grad[i], self.kernels[i,j], "full")

        # update kernels.biases using gradient descent 
            self.kernels -= learning_rate * kernels_grad
            self.biases -= learning_rate * output_grad

            return input_grad    
    
