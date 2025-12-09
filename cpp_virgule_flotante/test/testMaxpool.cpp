#include <iostream>
#include "../include/maxpool.hpp"

int main() {
    constexpr std::size_t C = 2;
    constexpr std::size_t H = 6;
    constexpr std::size_t W = 6;
    constexpr std::size_t out_rows = 3;
    constexpr std::size_t out_cols = 3;

    matrix3D<C, H, W> input = {
        {
            {1, 2, 3, 4, 5, 6},
            {5, 6, 7, 8, 9, 10},
            {9, 1, 2, 3, 4, 11},
            {4, 5, 6, 7, 8, 1},
            {0, 1, 2, 3, 4, 4},
            {4, 5, 6, 7, 8, 1}
        },
        {
            {1, 2, 3, 4, 5, 6},
            {5, 6, 7, 8, 9, 10},
            {9, 1, 2, 3, 4, 11},
            {4, 5, 6, 7, 8, 1},
            {0, 1, 2, 3, 4, 4},
            {4, 5, 6, 7, 8, 1}
        }
    };

    matrix3D<C, out_rows, out_cols> output;

    maxpool<C, H, W, 3, 2, out_rows, out_cols>(input, output);

    std::cout << "\nMaxpool output:\n";

    for (std::size_t ch = 0; ch < C; ++ch) {
        for (std::size_t i = 0; i < out_rows; ++i) {
            for (std::size_t j = 0; j < out_cols; ++j) {
                std::cout << output[ch][i][j] << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }

}
