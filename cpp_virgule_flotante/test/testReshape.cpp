#include <iostream>
#include "../include/reshape.hpp"

int main() {
    matrix3D<20,3,3> mat;

    // Fill matrix with deterministic values: mat[a][b][c] = a*100 + b*10 + c
    for (int a = 0; a < 20; ++a) {
        for (int b = 0; b < 3; ++b) {
            for (int c = 0; c < 3; ++c) {
                mat[a][b][c] = a*100 + b*10 + c;
                std::cout << a*100 + b*10 + c << " ";
            }
        std::cout << "\n";
        }
    std::cout << "\n";
    }

    std::cout << "generated input matrix \n";

    // Call reshape
    data_t out_vector[180];

    reshape(mat, out_vector);

    std::cout << "\n";
    for (std::size_t i = 0; i < 180; ++i) {
        std::cout << " " << out_vector[i];
    }
    std::cout << "\n";

    return 0;
}

