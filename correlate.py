import numpy as np

def correlate2d(input_array, kernel, mode='valid'):
    """
    Perform 2D correlation between an input array and a kernel.

    Parameters:
    input_array (numpy.ndarray): The input array
    kernel (numpy.ndarray): The kernel
    mode (str): The mode of correlation. Can be 'full' or 'valid'. Default is 'valid'.

    Returns:
    numpy.ndarray: The output array after correlation
    """
    # Get dimensions of input array and kernel
    input_rows, input_cols = input_array.shape
    kernel_rows, kernel_cols = kernel.shape
    
    # Determine the dimensions of the output array based on the mode
    if mode == 'full':
        output_rows = input_rows + kernel_rows - 1
        output_cols = input_cols + kernel_cols - 1
        # Pad the input array with zeros
        padded_input = np.pad(input_array, ((kernel_rows - 1, kernel_rows - 1), (kernel_cols - 1, kernel_cols - 1)), mode='constant')
    elif mode == 'valid':
        output_rows = input_rows - kernel_rows + 1
        output_cols = input_cols - kernel_cols + 1
        padded_input = input_array
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    
    # Initialize the output array
    output_array = np.zeros((output_rows, output_cols))
    
    # Slide the kernel over the input array and compute the sum of element-wise products at each position
    for i in range(output_rows):
        for j in range(output_cols):
            # Extract the region of the input array that overlaps with the kernel at this position
            region = padded_input[i:i + kernel_rows, j:j + kernel_cols]
            # Compute the sum of element-wise products and store the result in the output array
            output_array[i, j] = np.sum(region * kernel)
    return output_array

def convolve2d(input_array, kernel, mode='valid'):
    """
    Perform 2D convolution between an input array and a kernel.

    Parameters:
    input_array (numpy.ndarray): The input array
    kernel (numpy.ndarray): The kernel
    mode (str): The mode of convolution. Can be 'full' or 'valid'. Default is 'valid'.

    Returns:
    numpy.ndarray: The output array after convolution
    """
    # Flip the kernel
    kernel = np.flip(kernel)
    # Call the correlate2d function with the flipped kernel
    return correlate2d(input_array, kernel, mode)
