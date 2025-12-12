import cnn 
import image_reader
import read_coeff
import sys
import matplotlib.pyplot as plt
import image_crop_and_normalize as icn
import copy

path = '../cifar-10-binary/cifar-10-batches-bin/test_batch.bin'
images = image_reader.read_batch_file(path)
coefficients = list(
    read_coeff.read_coefficients('../cifar10_coeffs/CNN_coeff_3x3.txt', as_numpy=False)
)

# Cnn in all images make the error rate the index is the label 
# Make a flag that says we process every image or just one image --all
# Make a way to chose number of tested images from command line argument
all_images = '--all' in sys.argv
show_out = '--show' in sys.argv
nb = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 0
# Get Number of images by command line argument if --all is set
if all_images:
    if len(sys.argv) > 2 and sys.argv[2].isdigit():
        number_of_images = int(sys.argv[2])
    else:
        number_of_images = 10000  # Default to 10000 if not specified
    nb = min(number_of_images, len(images))

if all_images:
    progress_interval = nb // 20
    print("Processing {nb} images...".format(nb=nb))
    error = 0
    count = 0
    for image in images:
        coefs = copy.deepcopy(coefficients)
        cropped_normalized_image = icn.crop_and_normalize_image(image)
        output = cnn.cnn(cropped_normalized_image, coefs)
        predicted_label = output.index(max(output))
        #print(f"Processing image {count+1} with true label {image.label} and predicted label {predicted_label}")
        #print(f"CNN output probabilities: {output}")
        if predicted_label != image.label:
            error += 1
        count += 1
        if count % progress_interval == 0:
            print(f"Processed {count} / {nb} images...")
        if count == nb:
            break
    print(f"Total images: {count}, Errors: {error}, Error Rate: {error/count*100:.2f}% Success Rate: {(count - error)/count*100:.2f}%")
else:
    # Print only original image
    cropped_normalized_image = icn.crop_and_normalize_image(images[nb])
    image = images[nb]
    output = cnn.cnn(cropped_normalized_image, coefficients)
    if show_out == True:
        print("CNN output:", output)
        fig, axes = plt.subplots(1, 1, figsize=(4, 4))
        axes.imshow(image.pixels)
        axes.set_title(f'Label: {image.label} (Original)')
        axes.axis('off')
        plt.tight_layout()
        plt.show()