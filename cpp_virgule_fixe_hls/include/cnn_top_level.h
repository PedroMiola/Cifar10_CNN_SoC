#ifndef CNN_TOP_LEVEL_HPP
#define CNN_TOP_LEVEL_HPP

#include "../include/cnn.hpp"
#include "../include/convolution.hpp"
#include "../include/cnn_coefficients.hpp"
#include "../include/maxpool.hpp"
#include "../include/reshape.hpp"
#include "../include/perceptron.hpp"

void firstLayer(
    const image_t<IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH>& input_image,
    matrix3D<64, 12, 12>& output_feature_maps
);

void secondLayer(
    const matrix3D<64, 12, 12>& input_feature_maps,
    matrix3D<32, 6, 6>& output_feature_maps
);

void thirdLayer(
    const matrix3D<32, 6, 6>& input_feature_maps,
    matrix3D<20, 3, 3>& output_feature_maps
);

void fourthLayer(
    const matrix3D<20, 3, 3>& input_feature_maps,
    data_t output_logits[10]
);

void cnn_top_level(
    const image_t<IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH>& input_image,
    data_t output_logits[10]
);

#endif // CNN_TOP_LEVEL_HPP