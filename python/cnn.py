import convolution as conv
import maxpool as mp
import reshape as rp
import perceptron as pc

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

def firstLayer(matrix, kernel, bias, size=3, stride=2):
    convoluted_image = conv.convolve3d_4d(matrix, kernel, bias)
    return  mp.maxpool(convoluted_image, size, stride)

def secondLayer(matrix, kernel, bias, size=3, stride=2):
    convoluted_image = conv.convolve3d_4d(matrix, kernel, bias)
    return mp.maxpool(convoluted_image, size, stride)

def thirdLayer(matrix, kernel, bias, size=3, stride=2):
    convoluted_image = conv.convolve3d_4d(matrix, kernel, bias)
    return mp.maxpool(convoluted_image, size, stride)

def fourthLayer(matrix, kernel, bias):
    reshaped = rp.reshape_to_1d(matrix)
    return pc.perceptron(reshaped, kernel, bias)

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