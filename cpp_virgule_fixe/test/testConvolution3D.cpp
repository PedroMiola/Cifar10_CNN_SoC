#include "../include/convolution.hpp"  // convolve3d_4d + matrix3D/matrix4D + data_t
#include "testUtils.hpp"

#include <fstream>
#include <cmath>
#include <string>
#include <filesystem>
#include <cstddef>
#include <iostream>

int main() {

    std::filesystem::create_directories("../log");
    std::ofstream log("../log/outputConvolution3D_4D.log");
    if (!log) {
        std::cerr << "Error: Unable to open ../log/outputConvolution3D_4D.log\n";
        return 1;
    }

    int failures = 0;

    // ---------- Test 0: Single in/out channel, 3x3 identity kernel, zero bias ----------
    {
        constexpr std::size_t InC   = 1;
        constexpr std::size_t OutC  = 1;
        constexpr std::size_t H     = 3;
        constexpr std::size_t W     = 3;
        constexpr std::size_t KH    = 3;
        constexpr std::size_t KW    = 3;

        matrix3D<InC, H, W> input{};
        for (std::size_t i = 0; i < H; ++i) {
            for (std::size_t j = 0; j < W; ++j) {
                input[0][i][j] = static_cast<data_t>(i * 10 + j + 1);
            }
        }

        matrix4D<KH, KW, InC, OutC> kernel{};
        for (std::size_t ki = 0; ki < KH; ++ki)
            for (std::size_t kj = 0; kj < KW; ++kj)
                for (std::size_t ic = 0; ic < InC; ++ic)
                    for (std::size_t oc = 0; oc < OutC; ++oc)
                        kernel[ki][kj][ic][oc] = static_cast<data_t>(0);

        // Identity kernel: center tap = 1
        kernel[1][1][0][0] = static_cast<data_t>(1);

        data_t bias[OutC] = { static_cast<data_t>(0) };

        matrix3D<OutC, H, W> output{};
        convolve3d_4d<InC, H, W, KH, KW, OutC>(input, kernel, bias, output);

        bool ok = true;
        for (std::size_t i = 0; i < H && ok; ++i) {
            for (std::size_t j = 0; j < W; ++j) {
                if (std::fabs(static_cast<double>(output[0][i][j] - input[0][i][j])) > 1e-5) {
                    ok = false;
                    break;
                }
            }
        }
        logExpect(ok, failures, log, "T0: Single-channel identity convolution + zero bias (output == input)");
    }

    // ---------- Test 1: Multi-channel (3->2) 1x1 kernels, zero bias ----------
    {
        constexpr std::size_t InC   = 3;
        constexpr std::size_t OutC  = 2;
        constexpr std::size_t H     = 2;
        constexpr std::size_t W     = 3;
        constexpr std::size_t KH    = 1;
        constexpr std::size_t KW    = 1;

        matrix3D<InC, H, W> input{};
        // Channel 0: all 1, channel 1: all 10, channel 2: all 100
        for (std::size_t i = 0; i < H; ++i) {
            for (std::size_t j = 0; j < W; ++j) {
                input[0][i][j] = static_cast<data_t>(1);
                input[1][i][j] = static_cast<data_t>(10);
                input[2][i][j] = static_cast<data_t>(100);
            }
        }

        matrix4D<KH, KW, InC, OutC> kernel{};
        // out0 = 1*ch0 + 1*ch1 + 1*ch2 = 111
        kernel[0][0][0][0] = static_cast<data_t>(1);
        kernel[0][0][1][0] = static_cast<data_t>(1);
        kernel[0][0][2][0] = static_cast<data_t>(1);

        // out1 = 1*ch0 + 2*ch1 + 3*ch2 = 321
        kernel[0][0][0][1] = static_cast<data_t>(1);
        kernel[0][0][1][1] = static_cast<data_t>(2);
        kernel[0][0][2][1] = static_cast<data_t>(3);

        data_t bias[OutC] = {
            static_cast<data_t>(0),
            static_cast<data_t>(0)
        };

        matrix3D<OutC, H, W> output{};
        convolve3d_4d<InC, H, W, KH, KW, OutC>(input, kernel, bias, output);

        bool ok = true;
        for (std::size_t i = 0; i < H && ok; ++i) {
            for (std::size_t j = 0; j < W; ++j) {
                data_t expected0 = static_cast<data_t>(111);
                data_t expected1 = static_cast<data_t>(321);
                if (std::fabs(static_cast<double>(output[0][i][j] - expected0)) > 1e-5 ||
                    std::fabs(static_cast<double>(output[1][i][j] - expected1)) > 1e-5) {
                    ok = false;
                    break;
                }
            }
        }
        logExpect(ok, failures, log, "T1: 3 input channels combined into 2 outputs with 1x1 kernels");
    }

    // ---------- Test 2: Bias and ReLU behaviour ----------
    {
        constexpr std::size_t InC   = 1;
        constexpr std::size_t OutC  = 2;
        constexpr std::size_t H     = 2;
        constexpr std::size_t W     = 2;
        constexpr std::size_t KH    = 1;
        constexpr std::size_t KW    = 1;

        matrix3D<InC, H, W> input{};
        for (std::size_t i = 0; i < H; ++i)
            for (std::size_t j = 0; j < W; ++j)
                input[0][i][j] = static_cast<data_t>(1);

        matrix4D<KH, KW, InC, OutC> kernel{};
        // For both outputs, convolution sum is -2 per pixel
        kernel[0][0][0][0] = static_cast<data_t>(-2);
        kernel[0][0][0][1] = static_cast<data_t>(-2);

        // Bias[0] = 0 -> value = -2 -> ReLU -> 0
        // Bias[1] = +3 -> value = -2 + 3 = 1 -> ReLU -> 1
        data_t bias[OutC] = {
            static_cast<data_t>(0),
            static_cast<data_t>(3)
        };

        matrix3D<OutC, H, W> output{};
        convolve3d_4d<InC, H, W, KH, KW, OutC>(input, kernel, bias, output);

        bool ok = true;
        for (std::size_t i = 0; i < H && ok; ++i) {
            for (std::size_t j = 0; j < W; ++j) {
                data_t expected0 = static_cast<data_t>(0); // clamped by ReLU
                data_t expected1 = static_cast<data_t>(1); // negative + bias, then ReLU
                if (std::fabs(static_cast<double>(output[0][i][j] - expected0)) > 1e-5 ||
                    std::fabs(static_cast<double>(output[1][i][j] - expected1)) > 1e-5) {
                    ok = false;
                    break;
                }
            }
        }
        logExpect(ok, failures, log, "T2: Bias + ReLU: negative sums clamp to 0, bias can revive them");
    }

    // ---------- Test 3: 2 input channels -> 1 output, 3x3 identity per channel ----------
    {
        constexpr std::size_t InC   = 2;
        constexpr std::size_t OutC  = 1;
        constexpr std::size_t H     = 3;
        constexpr std::size_t W     = 4;
        constexpr std::size_t KH    = 3;
        constexpr std::size_t KW    = 3;

        matrix3D<InC, H, W> input{};

        // Channel 0
        data_t c0[H][W] = {
            {  1,  2,  3,  4 },
            {  5,  6,  7,  8 },
            {  9, 10, 11, 12 }
        };
        // Channel 1
        data_t c1[H][W] = {
            { 10, 20, 30, 40 },
            { 50, 60, 70, 80 },
            { 90,100,110,120 }
        };

        for (std::size_t i = 0; i < H; ++i) {
            for (std::size_t j = 0; j < W; ++j) {
                input[0][i][j] = c0[i][j];
                input[1][i][j] = c1[i][j];
            }
        }

        matrix4D<KH, KW, InC, OutC> kernel{};
        // Identity 3x3 kernel per input channel -> conv(channel) = channel
        for (std::size_t ki = 0; ki < KH; ++ki)
            for (std::size_t kj = 0; kj < KW; ++kj)
                for (std::size_t ic = 0; ic < InC; ++ic)
                    kernel[ki][kj][ic][0] = static_cast<data_t>(0);

        kernel[1][1][0][0] = static_cast<data_t>(1); // center on channel 0
        kernel[1][1][1][0] = static_cast<data_t>(1); // center on channel 1

        data_t bias[OutC] = { static_cast<data_t>(0) };

        matrix3D<OutC, H, W> output{};
        convolve3d_4d<InC, H, W, KH, KW, OutC>(input, kernel, bias, output);

        bool ok = true;
        for (std::size_t i = 0; i < H && ok; ++i) {
            for (std::size_t j = 0; j < W; ++j) {
                data_t expected = c0[i][j] + c1[i][j];
                if (std::fabs(static_cast<double>(output[0][i][j] - expected)) > 1e-5) {
                    ok = false;
                    break;
                }
            }
        }

        logExpect(ok, failures, log, "T3: 2-channel 3x3 identity kernels summed into 1 output channel");
    }

    // ---------- Test 4: Constant input, 2 in / 2 out, 3x3 kernels + bias + ReLU ----------
    {
        constexpr std::size_t InC   = 2;
        constexpr std::size_t OutC  = 2;
        constexpr std::size_t H     = 4;
        constexpr std::size_t W     = 4;
        constexpr std::size_t KH    = 3;
        constexpr std::size_t KW    = 3;

        matrix3D<InC, H, W> input{};

        // Channel 0 = 1, Channel 1 = 2 everywhere
        for (std::size_t i = 0; i < H; ++i) {
            for (std::size_t j = 0; j < W; ++j) {
                input[0][i][j] = static_cast<data_t>(1);
                input[1][i][j] = static_cast<data_t>(2);
            }
        }

        matrix4D<KH, KW, InC, OutC> kernel{};
        // out0: only channel 0 with all-ones 3x3 kernel (sum = 9), channel 1 = 0
        // out1: both channels with all-ones 3x3 (each sum = 9)
        for (std::size_t ki = 0; ki < KH; ++ki) {
            for (std::size_t kj = 0; kj < KW; ++kj) {
                kernel[ki][kj][0][0] = static_cast<data_t>(1); // ch0 -> out0
                kernel[ki][kj][1][0] = static_cast<data_t>(0); // ch1 -> out0
                kernel[ki][kj][0][1] = static_cast<data_t>(1); // ch0 -> out1
                kernel[ki][kj][1][1] = static_cast<data_t>(1); // ch1 -> out1
            }
        }

        // For any position:
        //   out0 pre-bias = 1 * 9 + 2 * 0 = 9;   + bias[0]=-10 -> -1 -> ReLU -> 0
        //   out1 pre-bias = 1 * 9 + 2 * 9 = 27;  + bias[1]=-7  -> 20 -> ReLU -> 20
        data_t bias[OutC] = {
            static_cast<data_t>(-10),
            static_cast<data_t>(-7)
        };

        matrix3D<OutC, H, W> output{};
        convolve3d_4d<InC, H, W, KH, KW, OutC>(input, kernel, bias, output);

        bool ok = true;
        for (std::size_t i = 0; i < H && ok; ++i) {
            for (std::size_t j = 0; j < W; ++j) {
                data_t expected0 = static_cast<data_t>(0);
                data_t expected1 = static_cast<data_t>(20);
                if (std::fabs(static_cast<double>(output[0][i][j] - expected0)) > 1e-5 ||
                    std::fabs(static_cast<double>(output[1][i][j] - expected1)) > 1e-5) {
                    ok = false;
                    break;
                }
            }
        }

        logExpect(ok, failures, log,
                  "T4: Constant input, 2-in/2-out 3x3 kernels with bias + ReLU (one map zeros, one positive)");
    }

    // ---------- Summary ----------
    if (failures == 0) log << "All convolve3d_4d tests PASSED.\n";
    else               log << failures << " convolve3d_4d test(s) FAILED.\n";

    log.close();
    return failures;
}
