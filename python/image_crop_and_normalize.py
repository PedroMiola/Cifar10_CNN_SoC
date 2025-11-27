from image_reader import Image

def crop_and_normalize_image(image):
    cropped_pixels = [row[4:28] for row in image.pixels[4:28]]
    
    # Normalize pixel values
    normalized_pixels = []
    number_of_pixels = 24 * 24
    normalized_pixels_red_pixels = []
    normalized_pixels_green_pixels = []
    normalized_pixels_blue_pixels = []
    
    for color in range(3):
        channel_sum = sum(cropped_pixels[row][col][color] for row in range(24) for col in range(24))
        channel_mean = channel_sum / number_of_pixels
        channel_squared_diff_sum = sum((cropped_pixels[row][col][color] - channel_mean) ** 2 for row in range(24) for col in range(24))
        channel_stddev = (channel_squared_diff_sum / number_of_pixels) ** 0.5
        channel_div = max(channel_stddev, 1.0 / (number_of_pixels ** 0.5))
        print(f"Channel {color}: mean={channel_mean}, stddev={channel_stddev}, div={channel_div}")
        if color == 0:
            normalized_pixels_red_pixels = [(cropped_pixels[row][col][0] - channel_mean) / channel_div for row in range(24) for col in range(24)]
        elif color == 1:
            normalized_pixels_green_pixels = [(cropped_pixels[row][col][1] - channel_mean) / channel_div for row in range(24) for col in range(24)]
        else:
            normalized_pixels_blue_pixels = [(cropped_pixels[row][col][2] - channel_mean) / channel_div for row in range(24) for col in range(24)]

    for i in range(24):
        row = []
        for j in range(24):
            r = normalized_pixels_red_pixels[i * 24 + j]
            g = normalized_pixels_green_pixels[i * 24 + j]
            b = normalized_pixels_blue_pixels[i * 24 + j]
            row.append((r, g, b))
        normalized_pixels.append(row)
    return Image(image.label, normalized_pixels)

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