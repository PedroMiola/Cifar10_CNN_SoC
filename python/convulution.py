# Receives a 2D matrix and a kernel, performs convolution, and returns the resulting matrix without numpy
def convolve2d(matrix, kernel):
    kernel_height = len(kernel)
    kernel_width = len(kernel[0])
    matrix_height = len(matrix)
    matrix_width = len(matrix[0])
    
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    
    output_matrix = [[0] * matrix_width for _ in range(matrix_height)]
    for row in range(matrix_height):
        for col in range(matrix_width):
            conv_sum = 0
            for ki in range(kernel_height):
                for kj in range(kernel_width):
                    mi = row + ki - pad_height
                    mj = col + kj - pad_width
                    if 0 <= mi < matrix_height and 0 <= mj < matrix_width:
                        conv_sum += matrix[mi][mj] * kernel[ki][kj]
            output_matrix[row][col] = conv_sum 
    return output_matrix    
            

# Apply a simple edge detection convolution to an image's pixel data
def apply_convolution_to_image(image):
    edge_detection_kernel = [
        [0, -1, 0],
        [-1,  5, -1],
        [0, -1, 0]
    ]
    
    height = len(image.pixels)
    width = len(image.pixels[0])
    convoluted_pixels = [[(0, 0, 0) for _ in range(width)] for _ in range(height)]
    
    for channel in range(3):
        channel_matrix = [[image.pixels[i][j][channel] for j in range(width)] for i in range(height)]
        
        convoluted_channel = convolve2d(channel_matrix, edge_detection_kernel)
        
        for i in range(height):
            for j in range(width):
                value = convoluted_channel[i][j]
                r, g, b = convoluted_pixels[i][j]
                if channel == 0:
                    convoluted_pixels[i][j] = (value, g, b)
                elif channel == 1:
                    convoluted_pixels[i][j] = (r, value, b)
                else:
                    convoluted_pixels[i][j] = (r, g, value)
    
    return convoluted_pixels

def apply_indenty_matrix(image):
    identity_kernel = [
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ]
    
    height = len(image.pixels)
    width = len(image.pixels[0])
    
    identity_pixels = [[(0, 0, 0) for _ in range(width)] for _ in range(height)]
    
    for channel in range(3):
        channel_matrix = [[image.pixels[i][j][channel] for j in range(width)] for i in range(height)]
        
        identity_channel = convolve2d(channel_matrix, identity_kernel)
        
        for i in range(height):
            for j in range(width):
                value = identity_channel[i][j]
                r, g, b = identity_pixels[i][j]
                if channel == 0:
                    identity_pixels[i][j] = (value, g, b)
                elif channel == 1:
                    identity_pixels[i][j] = (r, value, b)
                else:
                    identity_pixels[i][j] = (r, g, value)
    
    return identity_pixels
