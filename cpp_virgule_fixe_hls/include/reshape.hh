#ifndef RES_HPP
#define RES_HPP

#include "../include/cnn.hh"

inline void reshape(const matrix3D<20, 3, 3>& matrix , data_t(&flattened)[180]) { 
    for (int i=0; i<3; ++i) { 
        for (int j=0; j<3; ++j) { 
            for (int k=0; k<20; ++k) { 
                flattened[i*3*20 + j*20 + k] = matrix[k][i][j];
            }
        }
    } 
}

#endif