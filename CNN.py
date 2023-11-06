import numpy as np 
from layer import layer 
from scipy import signal 

class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        # TODO: return output
        pass

    def backward(self, output_gradient, learning_rate):
        # TODO: update parameters and return input gradient
        pass


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
        '''
        
        '''
        # when function called and input is given, save the input for the conv. class to remember
        self.input = input

        # copies the biases to output 
        self.output = np.copy(self.biases)


        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += signal.correlate2d

    def backward(self, output_grad, learning_rate):
        # given the shape of the matrix, establish a similar sized empty gradient matrix 
        kernels_grad = np.zeros(self.kernels_shape) 
        input_grad = np.zeros(self.in_shape)        

        # using two for loops to compute the kernel gradient for each kernel of given depth
        # which is 
        # bias gradient doesn't need to be calculated since it is the same as the output gradient 
        for i in range(self.depth):
            for j in range(self.in_depth):
                kernels_grad[i,j] = signal.correlate2d(self.input[j], output_grad[i], "valid")

                # computing input gradient by convolving 
                input_grad[i,j] += signal.convolve2d(output_grad[i], self.kernels[i,j], "full")

        # update kernels.biases using gradient descent 
            self.kernels -= learning_rate * kernels_grad
            self.biases -= learning_rate * output_grad

            return input_grad

    class Reshape(Layer):
        def __init__(self, in_shape, out_shape):
            self.in_shape = in_shape
            self.out_shape = out_shape

        def forward(self, input):
            '''
            takes an input and reshapes it to output shape
            '''
            return np.reshape(input, self.out_shape)
        
        def backward(self, output_grad, learning_rate): 
            return np.reshape(output_grad, self.in_shape)
        


