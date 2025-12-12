#include "../include/cnn_coefficients.hpp"

#include <iostream>

int  main (){

    // Print bias1

    for (std::size_t i = 0; i < 64; ++i){
        std::cout << "Bias conv1[" << i << "] = " << conv1_biases[i] << std::endl;
    }

    //
    for (std::size_t i = 0; i < 10; ++i){
        for (std::size_t j = 0; j < 180; ++j){
            std::cout << "Perceptron weights[" << i << "][" << j << "] = " << local3_weights[j][i] << std::endl;
        }
    }

    return 0;
}