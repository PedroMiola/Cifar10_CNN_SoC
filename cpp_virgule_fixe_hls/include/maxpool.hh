#ifndef MAX_HPP
#define MAX_HPP

#include "../include/cnn.hh"

template<   int channels, 
            int rows, 
            int cols,
            int size = 3, 
            int stride = 2,
            int out_rows, 
            int out_cols
>

void maxpool(matrix3D<channels, rows, cols> matrix, matrix3D<channels, out_rows, out_cols> &out_mtx) {

    for (int c = 0; c < channels; c++) { // once for every channel

        for (int i = 0; i < out_rows; i++) { // maxpool this channel
            for (int j = 0; j < out_cols; j++) { // 3x3 window, corresponding to a value in the final matrix

                int start_r = i * stride;
                int start_c = j * stride;

                data_t local_max = 0;

                for (int m = 0; m < size; m++) { // search window on original matrix and take the maximum
                    for (int n = 0; n < size; n++) {
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
