import image_crop_and_normalize as icn
import image_reader
import sys

def save_all_cropped_and_normalized_images(input_path, output_path):
    images = image_reader.read_batch_file(input_path)
    icn.save_batch_of_cropped_and_normalized_images(images, output_path)

if __name__ == "__main__":
    input_path = '../cifar-10-binary/cifar-10-batches-bin/test_batch.bin'
    output_path = '../cifar-10-binary/cifar-10-batches-cropped-bin/test_batch.bin'
    #save_all_cropped_and_normalized_images(input_path, output_path)
    #print(f"All cropped and normalized images saved to {output_path}")
    # Load first image to verify
    loaded_images_normalized = icn.load_batch_of_cropped_and_normalized_images(output_path)
    loaded_images = image_reader.read_batch_file(input_path)
    # Get number of images to verify by command line so num_images can be limited
    num_images = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    for i in range(num_images):
        assert loaded_images_normalized[i].label == loaded_images[i].label, f"Label mismatch at index {i}"
        first_image_normalized = loaded_images_normalized[i]
        first_image = icn.crop_and_normalize_image(loaded_images[i])
    # Verify pixel values
        for row in range(24):
            for col in range(24):
                for channel in range(3):
                    val1 = first_image_normalized.pixels[row][col][channel]
                    val2 = first_image.pixels[row][col][channel]
                    assert abs(val1 - val2) < 1e-6, f"Pixel value mismatch at image {i}, row {row}, col {col}, channel {channel}: {val1} vs {val2}"
        print(f"Image {i+1} label: {first_image.label}")
        # Print first two pyxels of image
        print(f"Image {i+1} first two pyxels: R({first_image.pixels[0][0][0]}, {first_image.pixels[0][1][0]}), G({first_image.pixels[0][0][1]}, {first_image.pixels[0][1][1]}), B({first_image.pixels[0][0][2]}, {first_image.pixels[0][1][2]})")
                    
    print("Verification successful: The first loaded normalized image matches the original normalized image.")
