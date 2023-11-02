import numpy as np

def correlate2d(input_array, kernel):
    # Get dimensions of input array and kernel
    input_rows, input_cols = input_array.shape
    kernel_rows, kernel_cols = kernel.shape
    
    # Determine the dimensions of the output array based on the mode
    output_rows = input_rows - kernel_rows + 1
    output_cols = input_cols - kernel_cols + 1

    # Initialize the output array
    output_array = np.zeros((output_rows, output_cols))
    
    # Slide the kernel over the input array and compute the sum of element-wise products at each position
    for i in range(output_rows):
        for j in range(output_cols):
            # Extract the region of the input array that overlaps with the kernel at this position
            region = input_array[i:i + kernel_rows, j:j + kernel_cols]
            # Compute the sum of element-wise products and store the result in the output array
            output_array[i, j] = np.sum(region * kernel)
    
    return output_array

input_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
kernel = np.array([[1, 0], [0, -1]])
output_array = correlate2d(input_array, kernel)
print(output_array)
