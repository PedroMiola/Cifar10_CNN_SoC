#test the convultion function with the image_reader.py
from image_reader import read_batch_file
from convolution import apply_convolution_to_image, apply_indenty_matrix
import sys
import matplotlib.pyplot as plt

path = '../cifar-10-binary/cifar-10-batches-bin/data_batch_1.bin'
images = read_batch_file(path)
nb = int(sys.argv[1]) if len(sys.argv) > 1 else 0
first_image = images[nb]

# Apply convolution to the first image
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(first_image.pixels)
axes[0].set_title(f'Label: {first_image.label} (Original)')
axes[0].axis('off')

convoluted_pixels = apply_convolution_to_image(first_image)
axes[1].imshow(convoluted_pixels)
axes[1].set_title(f'Label: {first_image.label} (Convoluted)')
axes[1].axis('off')

convoluted_pixels_identity = apply_indenty_matrix(first_image)
axes[2].imshow(convoluted_pixels_identity)
axes[2].set_title(f'Label: {first_image.label} (Identity)')
axes[2].axis('off')

plt.tight_layout()
plt.show()

# Print convulted pixel values for verification
print("Original Pixels (first 5x5):")
for row in first_image.pixels[:5]:
    print(row[:5])
print("\nConvoluted Pixels (first 5x5):")
for row in convoluted_pixels[:5]:
    print(row[:5])
print("\nIdentity Convoluted Pixels (first 5x5):")
for row in convoluted_pixels_identity[:5]:
    print(row[:5])
