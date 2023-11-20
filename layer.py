"""
Based on 'Neural Network from Scratch' by TheIndependentCode.
Source: https://github.com/TheIndependentCode/Neural-Network/blob/master/layer.py
"""
class Layer:
    def __init__(self):
        # This method can be used to initialize layer-specific parameters and structures.
        self.input = None
        self.output = None

    def forward(self, input):
        raise NotImplementedError("Forward pass not implemented")

    def backward(self, output_gradient, learning_rate):
        raise NotImplementedError("Backward pass not implemented")
