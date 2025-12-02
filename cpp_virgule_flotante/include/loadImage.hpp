#ifndef LOADIMAGE_HPP
#define LOADIMAGE_HPP

#include "cnn.hpp"
#include <string>

void loadImagesFromFile(const std::string& filepath, LabeledImage<IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH>* images, std::size_t numImages=0);

#endif