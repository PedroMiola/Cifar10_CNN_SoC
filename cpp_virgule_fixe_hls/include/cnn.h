#ifndef CNN_HPP
#define CNN_HPP

#include <cstddef>
#include <cstdint>
#include <iomanip>
#include "lib/ac_fixed.h"
#include "lib/ac_int.h"

#define data_t ac_fixed<32,16,true>
#define label_t uint8_t
#define IMAGE_HEIGHT 24
#define IMAGE_WIDTH 24
#define IMAGE_CHANNELS 3

template<int Rows, int Cols>
using matrix2D = data_t[Rows][Cols];

template<int Depth, int Rows, int Cols>
using matrix3D = data_t[Depth][Rows][Cols];

template<int Blocks, int Depth, int Rows, int Cols>
using matrix4D = data_t[Blocks][Depth][Rows][Cols];

template<int Channels, int Height, int Width>
using image_t = matrix3D<Channels, Height, Width>;
template<int Channels, int Height, int Width>
struct LabeledImage {
    image_t<Channels, Height, Width> img;
    label_t label;
};

#endif // CNN_HPP