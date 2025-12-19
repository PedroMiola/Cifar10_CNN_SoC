#include "../include/loadImage.hpp"
#include <fstream>
#include <iostream>
#include <sstream>

int main() {
    std::string img_files = "../../cifar-10-binary/cifar-10-batches-cropped-bin/test_batch.bin";

    const std::size_t num_test_images = 10;

    LabeledImage<IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH>* test_images = new LabeledImage<IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH>[num_test_images];
    loadImagesFromFile(img_files, test_images, num_test_images);

    for(std::size_t i = 0; i < num_test_images; ++i){
        std::cout << "Image " << i << " Label: " << static_cast<int>(test_images[i].label) << std::endl;
        for (std::size_t c = 0; c < IMAGE_CHANNELS; ++c) {
            std::cout << "Channel " << c << ":" << std::endl;
            for (std::size_t h = 0; h < IMAGE_HEIGHT; ++h) {
                for (std::size_t w = 0; w < IMAGE_WIDTH; ++w) {
                    std::cout << test_images[i].img[c][h][w] << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
    }

    delete[] test_images;

    return 0;
}