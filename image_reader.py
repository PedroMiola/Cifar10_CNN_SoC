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

path = 'cifar-10-binary/cifar-10-batches-bin/data_batch_1.bin'
images = read_batch_file(path)

#Display the image using matplotlib recive image in command line argument
#import sys
#import matplotlib.pyplot as plt
#nb = int(sys.argv[1]) if len(sys.argv) > 1 else 0

#first_image_pixels = images[nb].pixels
#plt.imshow(first_image_pixels)
#plt.title(f'Label: {images[nb].label}')
#plt.axis('off')
#plt.show()