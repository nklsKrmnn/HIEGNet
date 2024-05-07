

def calc_conv_output_size(input_size: int, kernel_size: int, stride: int, padding: int):
    """
    Calculate the output size of a convolutional layer based on the input shape and the convolutional parameters.
    :param input_size: Size of the input (just one dimension)
    :param kernel_size: Kernel Size
    :param stride: Stride
    :param padding: Padding
    :return: Output size
    """
    return (input_size - kernel_size + 2 * padding) // stride + 1