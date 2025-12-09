#ifndef MAX_HPP
#define MAX_HPP

#include "../include/cnn.hpp"

template<std::size_t channels, std::size_t rows, std::size_t cols,
         std::size_t size = 3, std::size_t stride = 2,
         std::size_t out_rows, std::size_t out_cols>

void maxpool(matrix3D<channels, rows, cols> matrix, matrix3D<channels, out_rows, out_cols> &out_mtx) {

    for (std::size_t c = 0; c < channels; c++) { // once for every channel

        for (std::size_t i = 0; i < out_rows; i++) { // maxpool this channel
            for (std::size_t j = 0; j < out_cols; j++) { // 3x3 window, corresponding to a value in the final matrix

                std::size_t start_r = i * stride;
                std::size_t start_c = j * stride;

                data_t local_max = 0;

                for (std::size_t m = 0; m < size; m++) { // search window on original matrix and take the maximum
                    for (std::size_t n = 0; n < size; n++) {
                        if ((start_r + m < rows) && (start_c + n < cols)) { // if within bounds
                                if (matrix[c][start_r + m][start_c + n] > local_max) {  // if found a greater value
                                    local_max = matrix[c][start_r + m][start_c + n];
                                }
                        }
                    }
                }
                
                out_mtx[c][i][j] = local_max;
            }
        }

    }

}

#endif // MAX_HPP

