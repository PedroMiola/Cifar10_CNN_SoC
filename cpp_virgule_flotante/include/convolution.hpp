#ifndef CONVOLUTION_HPP
#define CONVOLUTION_HPP

#include "cnn.hpp"

// 2D convolution with mirror padding.
// - input  : [Height][Width]
// - kernel : [KHeight][KWidth]
// - output : [Height][Width]  (same spatial size as input)
template<std::size_t Height, std::size_t Width,
         std::size_t KHeight, std::size_t KWidth>
void convolve2d(
    const matrix2D<Height, Width>   &input,
    const matrix2D<KHeight, KWidth> &kernel,
    matrix2D<Height, Width>         &output
){
    constexpr int pad_h = static_cast<int>(KHeight) / 2;
    constexpr int pad_w = static_cast<int>(KWidth) / 2;

    for (std::size_t row = 0; row < Height; ++row) {
        for (std::size_t col = 0; col < Width; ++col) {

            data_t conv_sum = static_cast<data_t>(0); // float

            for (std::size_t ki = 0; ki < KHeight; ++ki) {
                for (std::size_t kj = 0; kj < KWidth; ++kj) {

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


// 3D-4D convolution + bias + ReLU, mirroring your Python `convolve3d_4d`.
// - input  : [InC][Height][Width]
// - kernel : [KHeight][KWidth][InC][OutC]
// - bias   : [OutC]
// - output : [OutC][Height][Width]  (same H, W as input)
template<std::size_t InC,
         std::size_t Height,
         std::size_t Width,
         std::size_t KHeight,
         std::size_t KWidth,
         std::size_t OutC>
void convolve3d_4d(
    const matrix3D<InC, Height, Width>     &input,
    const matrix4D<KHeight, KWidth, InC, OutC> &kernel,
    const data_t                            bias[OutC],
    matrix3D<OutC, Height, Width>         &output
);

inline data_t relu(data_t x) {return x > static_cast<data_t>(0) ? x : static_cast<data_t>(0);}
#endif // CONVOLUTION_HPP