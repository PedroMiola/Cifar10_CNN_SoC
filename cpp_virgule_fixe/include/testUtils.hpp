#ifndef TESTUTILS_HPP
#define TESTUTILS_HPP

#include "cnn.hpp"
#include <fstream>
#include <string>
#include <iostream>
#include <cmath>

void logExpect(bool cond, int& fails, std::ofstream& log, const std::string& msg) {
    if (cond) { log << "[ OK ] " << msg << std::endl; }
    else { log << "[FAIL] " << msg << std::endl; ++fails; }
}

template<std::size_t H, std::size_t W>
bool matricesEqual(const matrix2D<H, W>& a,
                   const matrix2D<H, W>& b,
                   double eps = 1e-5)
{
    for (std::size_t i = 0; i < H; ++i) {
        for (std::size_t j = 0; j < W; ++j) {
            if (std::fabs(static_cast<double>(a[i][j] - b[i][j])) > eps) {
                return false;
            }
        }
    }
    return true;
}

#endif // TESTUTILS_HPP