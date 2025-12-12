from multiprocessing.util import debug
import convolution as conv
import maxpool as mp
import reshape as rp
import perceptron as pc

debug_1st = False  # Set to True to enable debugging prints in first layer
debug_2nd = False  # Set to True to enable debugging prints in second layer
debug_3rd = False  # Set to True to enable debugging prints in third layer
debug_4th = False  # Set to True to enable debugging prints in fourth layer

def _get_shape(nested_list):
    shape = []
    current_level = nested_list
    while isinstance(current_level, list):
        shape.append(len(current_level))
        if len(current_level) > 0:
            current_level = current_level[0]
        else:
            break
    return tuple(shape)

def print_matrix_3d(mat):
    """
    Channel = column (j). Also prints with lines/columns swapped as requested,
    and formats each value with 6 decimal places.
    """
    in_c = len(mat)           # C
    h = len(mat[0])           # H
    w = len(mat[0][0])        # W

    for j in range(w):  # "channel" = column
        print(f"Channel {j}:")
        for c in range(in_c):
            for i in range(h):
                print(f"{float(mat[c][i][j]):.4f}", end=" ")
            print()
        print()




def firstLayer(matrix, kernel, bias, size=3, stride=2):
    if debug_1st == True:
        print("First Layer Input Matrix:")
        print_matrix_3d(matrix)
    convoluted_image = conv.convolve3d_4d(matrix, kernel, bias)
    if debug_1st == True:
        print("First Layer Convoluted Matrix:")
        print_matrix_3d(convoluted_image)
    pooled_image = mp.maxpool(convoluted_image, size, stride)
    if debug_1st == True:
        print("First Layer Pooled Matrix:")
        print_matrix_3d(pooled_image)
    return pooled_image

def secondLayer(matrix, kernel, bias, size=3, stride=2):
    if debug_2nd == True:
        print("Second Layer Input Matrix:")
        print_matrix_3d(matrix)
    convoluted_image = conv.convolve3d_4d(matrix, kernel, bias)
    if debug_2nd == True:
        print("Second Layer Convoluted Matrix:")
        print_matrix_3d(convoluted_image)
    pooled_image = mp.maxpool(convoluted_image, size, stride)
    if debug_2nd == True:
        print("Second Layer Pooled Matrix:")
        print_matrix_3d(pooled_image)
    return pooled_image

def thirdLayer(matrix, kernel, bias, size=3, stride=2):
    # Print matrix before convolution for debugging
    if debug_3rd == True:
        print("Third Layer Input Matrix:")
        print_matrix_3d(matrix)
    convoluted_image = conv.convolve3d_4d(matrix, kernel, bias)
    # Print matrix after convolution for debugging
    if debug_3rd == True:
        print("Third Layer Convoluted Matrix:")
        print_matrix_3d(convoluted_image)
    pooled_image = mp.maxpool(convoluted_image, size, stride)
    if debug_3rd == True:
        print("Third Layer Pooled Matrix:")
        print_matrix_3d(pooled_image)
    return pooled_image

def fourthLayer(matrix, kernel, bias):
    if debug_4th == True:
        print("Fourth Layer Reshape Start")
        print_matrix_3d(matrix)
    reshaped = rp.reshape_to_1d(matrix)
    if debug_4th == True:
        print("Fourth Layer Reshape End")
        for val in reshaped:
            print(f"{val:.4f}", end="\n")
        print()
    output = pc.perceptron(reshaped, kernel, bias)
    if debug_4th == True:
        print("Fourth Layer Perceptron End")
        for val in output:
            print(f"{val:.4f}", end=" ")
        print()
    return output

def cnn(image, coefficients):
    pixels = image.pixels
    first_layer_kernel = None
    first_layer_bias = None
    second_layer_kernel = None
    second_layer_bias = None
    third_layer_kernel = None
    third_layer_bias = None
    fourth_layer_kernel = None
    fourth_layer_bias = None
    for tensor_name, value in coefficients:
        if tensor_name == 'conv1/weights':
            first_layer_kernel = value
        elif tensor_name == 'conv1/biases':
            first_layer_bias = value
        elif tensor_name == 'conv2/weights':
            second_layer_kernel = value
        elif tensor_name == 'conv2/biases':
            second_layer_bias = value
        elif tensor_name == 'conv3/weights':
            third_layer_kernel = value
        elif tensor_name == 'conv3/biases':
            third_layer_bias = value
        elif tensor_name == 'local3/weights':
            fourth_layer_kernel = value
        elif tensor_name == 'local3/biases':
            fourth_layer_bias = value
        
    out1 = firstLayer(pixels, first_layer_kernel, first_layer_bias, size=3, stride=2)
    out2 = secondLayer(out1, second_layer_kernel, second_layer_bias, size=3, stride=2)
    out3 = thirdLayer(out2, third_layer_kernel, third_layer_bias, size=3, stride=2)
    out4 = fourthLayer(out3, fourth_layer_kernel, fourth_layer_bias)
    return out4