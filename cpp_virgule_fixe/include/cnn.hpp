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
using image_t = matrix3D<Channels, Height, Width>;
template<std::size_t Channels, std::size_t Height, std::size_t Width>
struct LabeledImage {
    image_t<Channels, Height, Width> img;
    label_t label;
};

// Print matrix functinon for debugging
template<std::size_t H, std::size_t W>
void printMatrix2D(const matrix2D<H, W>& mat) {
    for (std::size_t i = 0; i < H; ++i) {
        for (std::size_t j = 0; j < W; ++j) {
            std::cout << std::fixed << std::setprecision(6) << mat[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

template<std::size_t InC, std::size_t H, std::size_t W>
void printMatrix3D(const matrix3D<InC, H, W>& mat) {
    for (std::size_t c = 0; c < InC; ++c) {
        std::cout << "Channel " << c << ":\n";
        for (std::size_t i = 0; i < H; ++i) {
            for (std::size_t j = 0; j < W; ++j) {
                std::cout  << mat[c][i][j] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

template<std::size_t Blocks, std::size_t InC, std::size_t H, std::size_t W>
void printMatrix4D(const matrix4D<Blocks, InC, H, W>& mat) {
    for (std::size_t b = 0; b < Blocks; ++b) {
        std::cout << "Block " << b << ":\n";
        for (std::size_t c = 0; c < InC; ++c) {
            std::cout << " Channel " << c << ":\n";
            for (std::size_t i = 0; i < H; ++i) {
                for (std::size_t j = 0; j < W; ++j) {
                    std::cout << std::fixed << std::setprecision(6) << mat[b][c][i][j] << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

#endif // CNN_HPP