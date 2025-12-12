import image_crop_and_normalize
import image_reader
import matplotlib.pyplot as plt
import sys

nb = int(sys.argv[1]) if len(sys.argv) > 1 else 0
path = '../cifar-10-binary/cifar-10-batches-bin/test_batch.bin'
images = image_reader.read_batch_file(path)
first_image = images[nb]
output_path = '../cifar-10-binary/cifar-10-batches-bin/cropped_normalized_image.bin'
#image_crop_and_normalize.save_batch_of_cropped_and_normalized_images(images, output_path)
#normalized_images = image_reader.read_cropped_normalized_batch(output_path)
#first_image_normalized = normalized_images[nb]
first_image_normalized = image_crop_and_normalize.crop_and_normalize_image(first_image)

# Plot the original and cropped normalized images for comparison
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(first_image.pixels)
axes[0].set_title(f'Label: {first_image.label} (Original)')
axes[0].axis('off')
axes[1].imshow(first_image_normalized.pixels)
axes[1].set_title(f'Label: {first_image_normalized.label} (Cropped & Normalized)')
axes[1].axis('off')
plt.tight_layout()
plt.show()

# Print pixel values of the cropped and normalized image for verification
print("Cropped and Normalized Pixels (first 5x5):")
for row in first_image_normalized.pixels[:5]:
    print(row[:5])
