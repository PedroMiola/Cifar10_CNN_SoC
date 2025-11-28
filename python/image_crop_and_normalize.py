from image_reader import Image
import struct

def crop_and_normalize_image(image):
    cropped_pixels = [row[4:28] for row in image.pixels[4:28]]
    
    # Normalize pixel values
    normalized_pixels = []
    number_of_pixels = 24 * 24 * 3
    normalized_pixels_red_pixels = []
    normalized_pixels_green_pixels = []
    normalized_pixels_blue_pixels = []
    total_sum = sum(cropped_pixels[row][col][color] for row in range(24) for col in range(24) for color in range(3))
    total_mean = total_sum / number_of_pixels
    total_squared_diff_sum = sum((cropped_pixels[row][col][color] - total_mean) ** 2 for row in range(24) for col in range(24) for color in range(3))
    total_stddev = (total_squared_diff_sum / number_of_pixels) ** 0.5
    total_div = max(total_stddev, 1.0 / (number_of_pixels ** 0.5))
    #print(f"Overall: mean={total_mean}, stddev={total_stddev}, div={total_div}")
    
    for row in range(24):
        for col in range(24):
            r = (cropped_pixels[row][col][0] - total_mean) / total_div
            g = (cropped_pixels[row][col][1] - total_mean) / total_div
            b = (cropped_pixels[row][col][2] - total_mean) / total_div
            normalized_pixels_red_pixels.append(r)
            normalized_pixels_green_pixels.append(g)
            normalized_pixels_blue_pixels.append(b)
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
                    f.write(value)

def save_batch_of_cropped_and_normalized_images(images, file_path):
    with open(file_path, 'wb') as f:
        for image in images:
            normalized_image = crop_and_normalize_image(image)
            f.write(bytes([image.label]))
            for row in normalized_image.pixels:
                for pixel in row:
                    for channel_value in pixel:
                        f.write(struct.pack('<f', float(channel_value)))


def load_batch_of_cropped_and_normalized_images(file_path):
    images = []
    with open(file_path, 'rb') as f:
        while True:
            label_byte = f.read(1)
            if not label_byte:
                break
            label = label_byte[0]
            pixels = []
            for i in range(24):
                row = []
                for j in range(24):
                    r_bytes = f.read(4)
                    g_bytes = f.read(4)
                    b_bytes = f.read(4)
                    r = struct.unpack('<f', r_bytes)[0]
                    g = struct.unpack('<f', g_bytes)[0]
                    b = struct.unpack('<f', b_bytes)[0]
                    row.append((r, g, b))
                pixels.append(row)
            images.append(Image(label, pixels))
    return images