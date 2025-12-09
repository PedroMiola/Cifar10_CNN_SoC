#ifndef RES_HPP
#define RES_HPP

#include "../include/cnn.hpp"

void reshape(const matrix3D<20, 3, 3>& matrix , data_t(&flattened)[180]) { for (std::size_t i=0; i<180; ++i) { flattened[i] = matrix[i/9][(i/3)%3][i%3]; } }

#endif