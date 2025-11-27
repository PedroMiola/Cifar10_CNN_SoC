import convolution as conv
import image_reader
import read_coeff
import sys

nb = int(sys.argv[1]) if len(sys.argv) > 1 else 0
path = '../cifar-10-binary/cifar-10-batches-bin/data_batch_1.bin'
images = image_reader.read_batch_file(path)
first_image = images[nb]
coefficients = read_coeff.read_coefficients('../cifar10_coeffs/CNN_coeff_3x3.txt', as_numpy=False)
coff = None
bias = None
for tensor_name, value in coefficients:
    if tensor_name == 'conv1/weights':
        #print(f'Kernel shape: {read_coeff._get_shape(value)}')
        coff = value
    elif tensor_name == 'conv1/biases':
        #print(f'Bias shape: {read_coeff._get_shape(value)}')
        bias = value

print(f"Shape of first weight tensor: {read_coeff._get_shape(coefficients['conv1/weights'])}")
convoluted_image = conv.convolve3d_4d(first_image.pixels, coff, bias)
print("Shape of convoluted image:", read_coeff._get_shape(convoluted_image))
# Print convulted pixel values for verification if flag -p is given
if '-p' in sys.argv:
    print("Convoluted Pixels (first 5x5):")
    for row in convoluted_image[:5]:
        print(row[:5])