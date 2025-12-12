#include "../include/cnn_top_level.hpp"
#include "../include/loadImage.hpp"

#include <iostream>

int main() {

    std::string img_files = "../../cifar-10-binary/cifar-10-batches-cropped-bin/test_batch.bin";
    const std::size_t num_test_images = 1000;

    LabeledImage<IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH>* test_images = new LabeledImage<IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH>[num_test_images];
    loadImagesFromFile(img_files, test_images, num_test_images);

    std::size_t correct_predictions = 0;

    for(std::size_t i = 0; i < num_test_images; ++i){

        data_t output_logits[10];

        cnn_top_level(
            test_images[i].img,
            output_logits
        );

        std::size_t predicted_label = 0;
        data_t max_logit = output_logits[0];
        for (std::size_t j = 1; j < 10; ++j) {
            if (output_logits[j] > max_logit) {
                max_logit = output_logits[j];
                predicted_label = j;
            }
        }
        if (predicted_label == test_images[i].label) {
            ++correct_predictions;
        }   
        //std::cout << "Image " << i << ": True label = " << static_cast<int>(test_images[i].label)
        //          << ", Predicted label = " << predicted_label << std::endl;
    }

    // Calculate and print accuracy
    data_t accuracy = static_cast<data_t>(correct_predictions) / static_cast<data_t>(num_test_images);
    std::cout << "Accuracy: " << accuracy * 100.0 << "%" << std::endl;  

    return 0;
}