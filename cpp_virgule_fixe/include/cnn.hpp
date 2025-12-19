#ifndef CNN_HPP
#define CNN_HPP

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include "lib/ac_fixed.h"
#include "lib/ac_int.h"
#include "data.hpp"
    
#define label_t uint8_t
#define IMAGE_HEIGHT 24
#define IMAGE_WIDTH 24
#define IMAGE_CHANNELS 3

template<std::size_t Rows, std::size_t Cols>
using matrix2D = data_t[Rows][Cols];

template<std::size_t Depth, std::size_t Rows, std::size_t Cols>
using matrix3D = data_t[Depth][Rows][Cols];

template<std::size_t Blocks, std::size_t Depth, std::size_t Rows, std::size_t Cols>
using matrix4D = data_t[Blocks][Depth][Rows][Cols];

template<std::size_t Channels, std::size_t Height, std::size_t Width>
using image_t = pyxel_t[Channels][Height][Width];

template<std::size_t Channels, std::size_t Height, std::size_t Width>
struct LabeledImage {
    image_t<Channels, Height, Width> img;
    label_t label;
};

#endif // CNN_HPP