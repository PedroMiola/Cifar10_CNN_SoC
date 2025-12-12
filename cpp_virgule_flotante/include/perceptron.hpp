#ifndef PERCEPTRON_HPP
#define PERCEPTRON_HPP

#include "cnn.hpp"

inline void perceptron(const data_t input_vector[180], data_t output_vector[10], const data_t weights[180][10], const data_t biases[10]) {
    for (std::size_t j = 0; j < 10; ++j) { // for each output neuron
        data_t sum = biases[j];
        for (std::size_t i = 0; i < 180; ++i) { // for each input
            sum += input_vector[i] * weights[i][j];
        }
        output_vector[j] = sum;
    }
}

#endif // PERCEPTRON_HPP