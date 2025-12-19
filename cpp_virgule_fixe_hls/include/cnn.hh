#ifndef CNN_HPP
#define CNN_HPP

#include <ac_fixed.h>
#include <ac_int.h>

#define data_t ac_fixed<32,16,true>
#define label_t ac_int<4,false>
#define dim_t int
//#define pixel_t ac_fixed<8,0,false>

#define IMAGE_HEIGHT 24
#define IMAGE_WIDTH 24
#define IMAGE_CHANNELS 3

template<int Rows, int Cols>
using matrix2D = data_t[Rows][Cols];

template<int Depth, int Rows, int Cols>
using matrix3D = data_t[Depth][Rows][Cols];

template<int Blocks, int Depth, int Rows, int Cols>
using matrix4D = data_t[Blocks][Depth][Rows][Cols];

template<std::size_t Channels, std::size_t Height, std::size_t Width>
using image_t = pyxel_t[Channels][Height][Width];


#endif // CNN_HPP
