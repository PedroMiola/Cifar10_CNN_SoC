#include "../include/cnn_top_level.hh"
#include "../include/images.hh"

// Receive number of test images in command line argument
int main(int argc, char* argv[]) {

    std::size_t num_test_images = 3;


    std::size_t correct_predictions = 0;
    std::size_t count = 0;

    // Image 1
    {
        label_t predicted_label;
        cnn_top_level(
            image1_pixels,
            predicted_label
        );

        if(predicted_label == image1_label){
            correct_predictions++;
        }
        count++;
    }
    
    {
        label_t predicted_label;
        cnn_top_level(
            image2_pixels,
            predicted_label
        );

        if(predicted_label == image2_label){
            correct_predictions++;
        }
        count++;
    }
    {
        label_t predicted_label;
        cnn_top_level(
            image3_pixels,
            predicted_label
        );

        if(predicted_label == image3_label){
            correct_predictions++;
        }
        count++;
    }

    // Calculate and print accuracy
    float accuracy = static_cast<float>(correct_predictions) / static_cast<float>(num_test_images);
    std::cout << "Accuracy: " << accuracy * 100.0 << "%" << std::endl;  

    return 0;
}