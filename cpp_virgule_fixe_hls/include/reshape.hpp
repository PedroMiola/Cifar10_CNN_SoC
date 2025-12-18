#ifndef RES_HPP
#define RES_HPP

#include "../include/cnn.hpp"

inline void reshape(const matrix3D<20, 3, 3>& matrix , data_t(&flattened)[180]) { 
    for (std::size_t i=0; i<3; ++i) { 
        for (std::size_t j=0; j<3; ++j) { 
            for (std::size_t k=0; k<20; ++k) { 
                flattened[i*3*20 + j*20 + k] = matrix[k][i][j];
            }
        }
    } 
}

#endif