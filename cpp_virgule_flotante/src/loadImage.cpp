#include "../include/loadImage.hpp"
#include <iostream>
#include <fstream>

void loadImagesFromFile(const std::string& filepath, LabeledImage<IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH>* images, std::size_t numImages) {
    // Open file in binary mode
    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Unable to open file " << filepath << std::endl;
        return;
    }

    // Read images until end of file or until numImages is reached
    std::size_t count = 0;
    while (numImages == 0 || count < numImages) {
        label_t label;
        data_t imgRedData[IMAGE_HEIGHT][IMAGE_WIDTH];
        data_t imgGreenData[IMAGE_HEIGHT][IMAGE_WIDTH];
        data_t imgBlueData[IMAGE_HEIGHT][IMAGE_WIDTH];
        // Read label
        file.read(reinterpret_cast<char*>(&label), sizeof(label_t));
        // Read image data pyxel by pixel
        for(std::size_t h = 0; h < IMAGE_HEIGHT; ++h) {
            for(std::size_t w = 0; w < IMAGE_WIDTH; ++w) {
                file.read(reinterpret_cast<char*>(&imgRedData[h][w]), sizeof(data_t));
                file.read(reinterpret_cast<char*>(&imgGreenData[h][w]), sizeof(data_t));
                file.read(reinterpret_cast<char*>(&imgBlueData[h][w]), sizeof(data_t));
            }
        }

        // If images pointer is provided, store the image
        if (images != nullptr) {
            images[count].label = label;
            for(std::size_t h = 0; h < IMAGE_HEIGHT; ++h){
                for(std::size_t w = 0; w < IMAGE_WIDTH; ++w) {
                    images[count].img[0][h][w] = imgRedData[h][w];
                    images[count].img[1][h][w] = imgGreenData[h][w];
                    images[count].img[2][h][w] = imgBlueData[h][w];
                }
            }
        } else {
            // If images is nullptr give error message
            std::cerr << "Warning: images pointer is nullptr, image data not stored." << std::endl;
        }
        ++count;
    }
}