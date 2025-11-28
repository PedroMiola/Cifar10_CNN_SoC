#ifndef LOADIMAGE_HPP
#define LOADIMAGE_HPP

#include "cnn.hpp" 
LabeledImage<IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH> loadImage(const std::string& filepath);

#endif