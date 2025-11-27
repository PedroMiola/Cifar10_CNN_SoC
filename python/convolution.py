def relu(x):
    if x < 0:
        return 0
    return x

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
            

def convolve3d_4d(matrix, kernel, bias):
    m_height = len(matrix)
    m_width = len(matrix[0])
    m_depth = len(matrix[0][0])
    k_n = len(kernel)
    k_m = len(kernel[0])
    k_l = len(kernel[0][0])
    k_c = len(kernel[0][0][0])
    # Output matrix shape will be (m_height, m_width, k_c)
    output_matrix = [[[0 for _ in range(k_c)] for _ in range(m_width)] for _ in range(m_height)]

    for c in range(k_c):
        matrix_sum = [[0 for _ in range(m_width)] for _ in range(m_height)] 
        for channel in range(m_depth):
            channel_matrix = [[matrix[i][j][channel] for j in range(m_width)] for i in range(m_height)]
            # Kernal shape is (k_n, k_m, k_l, k_c) we want to convolve with the kernal that the height and width are k_n and k_m 
            convoluted_channel = convolve2d(channel_matrix, [[kernel[i][j][channel][c] for j in range(k_m)] for i in range(k_n)])
            for i in range(m_height):
                for j in range(m_width):
                    matrix_sum[i][j] += convoluted_channel[i][j]
        
        for i in range(m_height):
            for j in range(m_width):
                matrix_sum[i][j] += bias[c]
                output_matrix[i][j][c] = relu(matrix_sum[i][j])
                
    return output_matrix

        


def apply_convolution_to_image(image, kernel):
    height = len(image.pixels)
    width = len(image.pixels[0])
    convoluted_pixels = [[(0, 0, 0) for _ in range(width)] for _ in range(height)]
    
    for channel in range(3):
        channel_matrix = [[image.pixels[i][j][channel] for j in range(width)] for i in range(height)]
        
        convoluted_channel = convolve2d(channel_matrix, kernel)
        
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

# Apply a simple edge detection convolution to an image's pixel data
def apply_convolution_to_image(image):
    #edge_detection_kernel = [
    #    [1/16, 2/16, 1/16],
    #    [2/16,  4/16, 2/16],
    #    [1/16, 2/16, 1/16]
    #]
    # 5x5 Gaussian blur kernel
    gaussian_blur_kernel = [
        [1/273, 4/273, 6/273, 4/273, 1/273],
        [4/273,16/273,24/273,16/273, 4/273],
        [6/273,24/273,36/273,24/273, 6/273],
        [4/273,16/273,24/273,16/273, 4/273],
        [1/273, 4/273, 6/273, 4/273, 1/273]
    ]
    
    height = len(image.pixels)
    width = len(image.pixels[0])
    convoluted_pixels = [[(0, 0, 0) for _ in range(width)] for _ in range(height)]
    
    for channel in range(3):
        channel_matrix = [[image.pixels[i][j][channel] for j in range(width)] for i in range(height)]
        
        convoluted_channel = convolve2d(channel_matrix, gaussian_blur_kernel)
        
        for i in range(height):
            for j in range(width):
                value = convoluted_channel[i][j]
                # Check if value is floating point and convert to int
                if isinstance(value, float):
                    value = int(value)       
                r, g, b = convoluted_pixels[i][j]
                if channel == 0:
                    convoluted_pixels[i][j] = (value, g, b)
                elif channel == 1:
                    convoluted_pixels[i][j] = (r, value, b)
                else:
                    convoluted_pixels[i][j] = (r, g, value)
    
    return convoluted_pixels

def apply_indenty_matrix(image):

    #identity_kernel = [
    #    [0, 0, 0],
    #    [0, 1, 0],
    #    [0, 0, 0]
    #]
    # A 9x9 identity kernel
    identity_kernel = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0]
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
