# Image has a label and a 2D array of pixels (32x32)
class Image:
    def __init__(self, label, pixels):
        self.label = label
        self.pixels = pixels  
    
def read_batch_file(file_path):
    # Reads each image from file each image is 3073 bytes
    with open(file_path, 'rb') as f:
        byte_data = f.read()
    image_size = 3073
    num_images = len(byte_data) // image_size
    images = []
    # Each pixel is represented by three byetes (R, G, B), the first 1024 bytes are red, next 1024 green, last 1024 blue in the file
    for i in range(num_images):
        start = i * image_size
        label = byte_data[start]
        red = byte_data[start + 1:start + 1025]
        green = byte_data[start + 1025:start + 2049]
        blue = byte_data[start + 2049:start + 3073]
        
        # Construct the 32x32 pixel array
        pixels = []
        for row in range(32):
            pixel_row = []
            for col in range(32):
                r = red[row * 32 + col]
                g = green[row * 32 + col]
                b = blue[row * 32 + col]
                pixel_row.append((r, g, b))
            pixels.append(pixel_row)
        
        images.append(Image(label, pixels))
    return images

def read_cropped_normalized_batch(file_path):
    with open(file_path, 'rb') as f:
        byte_data = f.read()
    image_size = 1729  
    num_images = len(byte_data) // image_size
    images = []

    for i in range(num_images):
        start = i * image_size
        label = byte_data[start]
        red = byte_data[start + 1:start + 577]
        green = byte_data[start + 577:start + 1153]
        blue = byte_data[start + 1153:start + 1729]
        
        pixels = []
        for row in range(24):
            pixel_row = []
            for col in range(24):
                r = red[row * 24 + col]
                g = green[row * 24 + col]
                b = blue[row * 24 + col]
                pixel_row.append((r, g, b))
            pixels.append(pixel_row)
        
        images.append(Image(label, pixels))
    return images