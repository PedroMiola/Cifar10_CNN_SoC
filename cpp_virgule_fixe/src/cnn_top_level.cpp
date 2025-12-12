#include "../include/cnn_top_level.hpp"

const char debug_1st_layer = false;
const char debug_2nd_layer = false;
const char debug_3rd_layer = false;
const char debug_4th_layer = false;

void firstLayer(
    const image_t<IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH>& input_image,
    matrix3D<64, 12, 12>& output_feature_maps
){
    // Convolution
    if (debug_1st_layer) printf("First Layer Convolution Start\n");
    if (debug_1st_layer) printMatrix3D<IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH>(input_image); // Debug print
    matrix3D<64, IMAGE_HEIGHT, IMAGE_WIDTH> conv_output{};
    convolve3d_4d<IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH, 3, 3, 64>(
        input_image,
        conv1_weights,
        conv1_biases,
        conv_output
    );

    if (debug_1st_layer) printf("First Layer Convolution End\n");
    if (debug_1st_layer) printMatrix3D<64, IMAGE_HEIGHT, IMAGE_WIDTH>(conv_output); // Debug print
    // Max Pooling
    maxpool<64, IMAGE_HEIGHT, IMAGE_WIDTH, 3, 2, 12, 12>(
        conv_output,
        output_feature_maps
    );

    if (debug_1st_layer) printf("First Layer Max Pooling End\n");
    if (debug_1st_layer) printMatrix3D<64, 12, 12>(output_feature_maps); // Debug print
}

void secondLayer(
    const matrix3D<64, 12, 12>& input_feature_maps,
    matrix3D<32, 6, 6>& output_feature_maps
){
    // Convolution
    if (debug_2nd_layer) printf("Second Layer Convolution Start\n");
    matrix3D<32, 12, 12> conv_output{};
    if (debug_2nd_layer) printMatrix3D<64, 12, 12>(input_feature_maps); // Debug print
    convolve3d_4d<64, 12, 12, 3, 3, 32>(
        input_feature_maps,
        conv2_weights,
        conv2_biases,
        conv_output
    );
    if (debug_2nd_layer) printf("Second Layer Convolution End\n");
    if (debug_2nd_layer) printMatrix3D<32, 12, 12>(conv_output); // Debug print

    // Max Pooling
    maxpool<32, 12, 12, 3, 2, 6, 6>(
        conv_output,
        output_feature_maps
    );
    if (debug_2nd_layer) printf("Second Layer Max Pooling End\n");
    if (debug_2nd_layer) printMatrix3D<32, 6, 6>(output_feature_maps); // Debug print
}

void thirdLayer(
    const matrix3D<32, 6, 6>& input_feature_maps,
    matrix3D<20, 3, 3>& output_feature_maps
){
    // Convolution
    if (debug_3rd_layer) printf("Third Layer Convolution Start\n");
    matrix3D<20, 6, 6> conv_output{};
    if (debug_3rd_layer) printMatrix3D<32, 6, 6>(input_feature_maps); // Debug print
    convolve3d_4d<32, 6, 6, 3, 3, 20>(
        input_feature_maps,
        conv3_weights,
        conv3_biases,
        conv_output
    );
    if (debug_3rd_layer) printf("Third Layer Convolution End\n");
    if (debug_3rd_layer) printMatrix3D<20, 6, 6>(conv_output); // Debug print
    // Max Pooling
    maxpool<20, 6, 6, 3, 2, 3, 3>(
        conv_output,
        output_feature_maps
    );
    if (debug_3rd_layer) printf("Third Layer Max Pooling End\n");
    if (debug_3rd_layer) printMatrix3D<20, 3, 3>(output_feature_maps); // Debug print
}

void fourthLayer(
    const matrix3D<20, 3, 3>& input_feature_maps,
    data_t output_logits[10]
){
    data_t input_feature_vector[180];
    if (debug_4th_layer) {
        printf("Fourth Layer Reshape Start\n");
        printMatrix3D<20, 3, 3>(input_feature_maps); // Debug print
    }
    reshape(input_feature_maps, input_feature_vector);
    if (debug_4th_layer) {
        printf("Fourth Layer Reshape End\n");
        for (std::size_t i = 0; i < 180; ++i) {
            printf("%.4f\n", input_feature_vector[i]);
        }
        printf("\n");
    }
    perceptron(
        input_feature_vector,
        output_logits,
        local3_weights,
        local3_biases
    );
    if (debug_4th_layer) {
        printf("Fourth Layer Perceptron End\n");
        for (std::size_t i = 0; i < 10; ++i) {
            printf("%.4f ", output_logits[i]);
        }
        printf("\n");
    }
}

void cnn_top_level(
    const image_t<IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH>& input_image,
    data_t output_prob[10]
){
    matrix3D<64, 12, 12> layer1_output{};
    firstLayer(input_image, layer1_output);

    matrix3D<32, 6, 6> layer2_output{};
    secondLayer(layer1_output, layer2_output);

    matrix3D<20, 3, 3> layer3_output{};
    thirdLayer(layer2_output, layer3_output);

    fourthLayer(layer3_output, output_prob);
}