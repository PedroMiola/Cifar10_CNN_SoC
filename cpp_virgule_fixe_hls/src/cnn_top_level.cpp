#include "../include/cnn_top_level.hh"

void firstLayer(
    const image_t<IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH>& input_image,
    matrix3D<64, 12, 12>& output_feature_maps
){
    matrix3D<64, IMAGE_HEIGHT, IMAGE_WIDTH> conv_output{};
    convolve3d_4d_Image<IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH, 3, 3, 64>(
        input_image,
        conv1_weights,  
        conv1_biases,
        conv_output
    );

    maxpool<64, IMAGE_HEIGHT, IMAGE_WIDTH, 3, 2, 12, 12>(
        conv_output,
        output_feature_maps
    );
}

void secondLayer(
    const matrix3D<64, 12, 12>& input_feature_maps,
    matrix3D<32, 6, 6>& output_feature_maps
){

    matrix3D<32, 12, 12> conv_output{};
    convolve3d_4d<64, 12, 12, 3, 3, 32>(
        input_feature_maps,
        conv2_weights,
        conv2_biases,
        conv_output
    );

    // Max Pooling
    maxpool<32, 12, 12, 3, 2, 6, 6>(
        conv_output,
        output_feature_maps
    );
}

void thirdLayer(
    const matrix3D<32, 6, 6>& input_feature_maps,
    matrix3D<20, 3, 3>& output_feature_maps
){
    matrix3D<20, 6, 6> conv_output{};
    convolve3d_4d<32, 6, 6, 3, 3, 20>(
        input_feature_maps,
        conv3_weights,
        conv3_biases,
        conv_output
    );

    // Max Pooling
    maxpool<20, 6, 6, 3, 2, 3, 3>(
        conv_output,
        output_feature_maps
    );
}

void fourthLayer(
    const matrix3D<20, 3, 3>& input_feature_maps,
    data_t output_logits[10]
){
    data_t input_feature_vector[180];
    reshape(input_feature_maps, input_feature_vector);

    perceptron(
        input_feature_vector,
        output_logits,
        local3_weights,
        local3_biases
    );

}

void getMaxIndex(
    const data_t output_prob[10],
    label_t& output_label
){
    label_t max_index = 0;
    data_t max_value = output_prob[0];
    for (label_t i = 1; i < 10; ++i) {
        if (output_prob[i] > max_value) {
            max_value = output_prob[i];
            max_index = i;
        }
    }
    output_label = max_index;
}

void cnn_top_level(
    const image_t<IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH>& input_image,
    label_t &output_label
){
    matrix3D<64, 12, 12> layer1_output{};
    firstLayer(input_image, layer1_output);

    matrix3D<32, 6, 6> layer2_output{};
    secondLayer(layer1_output, layer2_output);

    matrix3D<20, 3, 3> layer3_output{};
    thirdLayer(layer2_output, layer3_output);

    data_t output_prob[10];
    fourthLayer(layer3_output, output_prob);

    getMaxIndex(output_prob, output_label);
}