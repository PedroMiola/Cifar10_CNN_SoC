#include "../include/cnn_top_level.hpp"
#include "../include/loadImage.hpp"

// Receive number of test images in command line argument
int main(int argc, char* argv[]) {

    std::string img_files = "../../cifar-10-binary/cifar-10-batches-cropped-bin/test_batch.bin";
    // Number of test images in the CIFAR-10 test batch in command line argument
    std::size_t num_test_images = 10000;
    if (argc > 1) {
        num_test_images = static_cast<std::size_t>(std::stoi(argv[1]));
    }
    
    // 5% of test images for printing purpose
    std::size_t num_print_images = num_test_images / 20;

    LabeledImage<IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH>* test_images = new LabeledImage<IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH>[num_test_images];
    loadImagesFromFile(img_files, test_images, num_test_images);

    std::size_t correct_predictions = 0;
    std::size_t count = 0;

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

        //std::cout << "Image " << i << ": True label = " << static_cast<int>(test_images[i].label) << ", Predicted label = " << predicted_label << std::endl;

        if (!(i % num_print_images)){
            //std::cout << "Processed " << ++count * (5) << "% of test images." << std::endl;
        }
    }

    // Calculate and print accuracy
    float accuracy = static_cast<float>(correct_predictions) / static_cast<float>(num_test_images);
    std::cout << "Accuracy: " << accuracy * 100.0 << "%" << std::endl;  

    return 0;
}