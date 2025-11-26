# Crop the 32x32 image to 24x24 and normalize pixel values 

def crop_and_normalize_image(image):
    cropped_pixels = [row[4:28] for row in image.pixels[4:28]]
    
    # Normalize pixel values to range [0, 1]
    normalized_pixels = []
    for channel in range(3):
        number_of_pixels = 24 * 24
        mean_value_of_channel = sum(cropped_pixels[i][j][channel] for i in range(24) for j in range(24)) / (number_of_pixels)
        stddev_value_of_channel = (sum((cropped_pixels[i][j][channel] - mean_value_of_channel) ** 2 for i in range(24) for j in range(24)) / (number_of_pixels)) ** 0.5
        value = max(stddev_value_of_channel, 1/(number_of_pixels ** 0.5))
        for i in range(24):
            if channel == 0:
                normalized_row = []
                normalized_pixels.append(normalized_row)
            for j in range(24):
                r, g, b = cropped_pixels[i][j]
                if channel == 0:
                    normalized_value = (r - mean_value_of_channel) / value
                    normalized_pixels[i].append([normalized_value, 0, 0])
                elif channel == 1:
                    normalized_value = (g - mean_value_of_channel) / value
                    normalized_pixels[i][j][1] = normalized_value
                else:
                    normalized_value = (b - mean_value_of_channel) / value
                    normalized_pixels[i][j][2] = normalized_value

                
                
    
    return normalized_pixels

def save_croped_and_normalized_image(image, file_path):
    normalized_pixels = crop_and_normalize_image(image)
    with open(file_path, 'wb') as f:
        f.write(bytes([image.label]))
        for channel in range(3):
            for row in normalized_pixels:
                for pixel in row:
                    value = int(pixel[channel] * 255)
                    f.write(bytes([value]))

def save_batch_of_cropped_and_normalized_images(images, file_path):
    with open(file_path, 'wb') as f:
        for image in images:
            normalized_pixels = crop_and_normalize_image(image)
            f.write(bytes([image.label]))
            for channel in range(3):
                for row in normalized_pixels:
                    for pixel in row:
                        value = int(pixel[channel] * 255)
                        f.write(bytes([value]))