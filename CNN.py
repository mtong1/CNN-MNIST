import numpy as np 
from layer import layer 
from scipy import signal 

class Convolutional(Layer):
    '''
    Class that defines the functions of a Convolutional Neural Network, including the forward propagation. 
    '''
    def __init__(self, in_shape, kernel_size, depth):
    '''
    Parameters of init required to define our CNN.
        kernel_size: size of the kernel, determined by user 
        in_shape: shape of the input, given by a tuple that defines the input depth, height, and width 
        depth: how many input matrices there are, also determines the number of kernels per layer 
    '''
    in_depth, in_height, in_width = in_shape # input shape is a tuple that defines each of these parameters in this order 
    self.depth = depth
    self.in_shape = in_shape 
    self.in_depth = in_depth 
    self.kernels_shape = (depth, in_depth, kernel_size, kernel_size)
    # last two are size of matrices in each kernel

    # instantiating randomized kernel values to start
    self.kernels = np.random.randm(*self.kernels_shape)

    # instantiating randomized bias values to start (bias will be the shape of the output)
    self.biases = np.random.randm(*self.output_shape)

    def forward_prop(self, input):
        self.output = np.copy(self.biases)
        for i in range(self.depth):
            for j in range(self.input_depth):

