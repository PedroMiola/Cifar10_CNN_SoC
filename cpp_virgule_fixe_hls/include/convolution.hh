#ifndef CONVOLUTION_HPP
#define CONVOLUTION_HPP

#include "cnn.hh"

inline data_t relu(data_t x) {return x > static_cast<data_t>(0) ? x : static_cast<data_t>(0);}

// 2D convolution with mirror padding.
// - input  : [Height][Width]
// - kernel : [KHeight][KWidth]
// - output : [Height][Width]  (same spatial size as input)
template<int Height, int Width,
         int KHeight, int KWidth>
void convolve2d(
    const matrix2D<Height, Width>   &input,
    const matrix2D<KHeight, KWidth> &kernel,
    matrix2D<Height, Width>         &output
){
    constexpr int pad_h = static_cast<int>(KHeight) / 2;
    constexpr int pad_w = static_cast<int>(KWidth) / 2;

    for (int row = 0; row < Height; ++row) {
        for (int col = 0; col < Width; ++col) {

            data_t conv_sum = static_cast<data_t>(0); // float

            for (int ki = 0; ki < KHeight; ++ki) {
                for (int kj = 0; kj < KWidth; ++kj) {

                    int mi = static_cast<int>(row) + static_cast<int>(ki) - pad_h;
                    int mj = static_cast<int>(col) + static_cast<int>(kj) - pad_w;

                    if (mi < 0) mi = 0; 
                    else if (mi >= static_cast<int>(Height)) mi = static_cast<int>(Height) - 1;
                    if (mj < 0) mj = 0;
                    else if (mj >= static_cast<int>(Width)) mj = static_cast<int>(Width) - 1;

                    conv_sum += input[mi][mj] * kernel[ki][kj];
                }
            }

            output[row][col] = conv_sum;
        }
    }
}

// 3D convolution with 4D kernel and bias + ReLU activation.
// - input  : [InC][Height][Width]
// - kernel : [KHeight][KWidth][InC][OutC]
// - bias   : [OutC]
// - output : [OutC][Height][Width]  (same H, W as input)
template<int InC,
         int Height,
         int Width,
         int KHeight,
         int KWidth,
         int OutC>
void convolve3d_4d(
    const matrix3D<InC, Height, Width>           &input,
    const matrix4D<KHeight, KWidth, InC, OutC>   &kernel,
    const data_t                                  bias[OutC],
    matrix3D<OutC, Height, Width>               &output
)
{
    // Temporary buffers
    matrix2D<Height, Width> channel_matrix;
    matrix2D<Height, Width> conv_result;
    matrix2D<Height, Width> matrix_sum;
    matrix2D<KHeight, KWidth> kernel_slice;

    // For each output channel
    for (int out_c = 0; out_c < OutC; ++out_c) {

        for (int i = 0; i < Height; ++i) {
            for (int j = 0; j < Width; ++j) {
                matrix_sum[i][j] = static_cast<data_t>(0);
            }
        }

        for (int in_c = 0; in_c < InC; ++in_c) {

            for (int i = 0; i < Height; ++i) {
                for (int j = 0; j < Width; ++j) {
                    channel_matrix[i][j] = input[in_c][i][j];
                }
            }

            for (int ki = 0; ki < KHeight; ++ki) {
                for (int kj = 0; kj < KWidth; ++kj) {
                    kernel_slice[ki][kj] = kernel[ki][kj][in_c][out_c];
                }
            }

            convolve2d<Height, Width, KHeight, KWidth>(
                channel_matrix, kernel_slice, conv_result
            );

            for (int i = 0; i < Height; ++i) {
                for (int j = 0; j < Width; ++j) {
                    matrix_sum[i][j] += conv_result[i][j];
                }
            }
        }

        for (int i = 0; i < Height; ++i) {
            for (int j = 0; j < Width; ++j) {
                data_t value = matrix_sum[i][j] + bias[out_c];
                output[out_c][i][j] = relu(value);
            }
        }
    }
}

template<std::size_t InC,
         std::size_t Height,
         std::size_t Width,
         std::size_t KHeight,
         std::size_t KWidth,
         std::size_t OutC>
         
void convolve3d_4d_Image(
    const image_t<IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH>      &input,
    const matrix4D<KHeight, KWidth, InC, OutC>   &kernel,
    const data_t                                  bias[OutC],
    matrix3D<OutC, Height, Width>               &output
)
{
    // Temporary buffers
    matrix2D<Height, Width> channel_matrix;
    matrix2D<Height, Width> conv_result;
    matrix2D<Height, Width> matrix_sum;
    matrix2D<KHeight, KWidth> kernel_slice;

    // For each output channel
    for (std::size_t out_c = 0; out_c < OutC; ++out_c) {

        for (std::size_t i = 0; i < Height; ++i) {
            for (std::size_t j = 0; j < Width; ++j) {
                matrix_sum[i][j] = static_cast<data_t>(0);
            }
        }

        for (std::size_t in_c = 0; in_c < InC; ++in_c) {

            for (std::size_t i = 0; i < Height; ++i) {
                for (std::size_t j = 0; j < Width; ++j) {
                    channel_matrix[i][j] = input[in_c][i][j];
                }
            }

            for (std::size_t ki = 0; ki < KHeight; ++ki) {
                for (std::size_t kj = 0; kj < KWidth; ++kj) {
                    kernel_slice[ki][kj] = kernel[ki][kj][in_c][out_c];
                }
            }

            convolve2d<Height, Width, KHeight, KWidth>(
                channel_matrix, kernel_slice, conv_result
            );

            for (std::size_t i = 0; i < Height; ++i) {
                for (std::size_t j = 0; j < Width; ++j) {
                    matrix_sum[i][j] += conv_result[i][j];
                }
            }
        }

        for (std::size_t i = 0; i < Height; ++i) {
            for (std::size_t j = 0; j < Width; ++j) {
                data_t value = matrix_sum[i][j] + bias[out_c];
                output[out_c][i][j] = relu(value);
            }
        }
    }
}

#endif // CONVOLUTION_HPP
