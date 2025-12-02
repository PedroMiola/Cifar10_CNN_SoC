#include "../include/testUtils.hpp"
#include "../include/cnn.hpp"

#include <type_traits>
#include <fstream>
#include <string>
#include <cstddef>
#include <iostream>

int main() {
    std::ofstream log("../log/outputCNN.log");
    if (!log) {
        std::cerr << "Error: Unable to open ../log/outputCNN.log\n";
        return 1;
    }

    int failures = 0;

    // ---------- Test 0: matrix2D size & indexing ----------
    {
        constexpr std::size_t R = 3, C = 5;
        matrix2D<R, C> m2{};

        // size check
        bool sizeOK = sizeof(m2) == R * C * sizeof(data_t);
        logExpect(sizeOK, failures, log, "matrix2D sizeof == R*C*sizeof(data_t)");

        // fill & index check
        for (std::size_t r = 0; r < R; ++r)
            for (std::size_t c = 0; c < C; ++c)
                m2[r][c] = static_cast<data_t>(r * 100 + c);

        bool valsOK = true;
        for (std::size_t r = 0; r < R && valsOK; ++r)
            for (std::size_t c = 0; c < C; ++c)
                if (m2[r][c] != static_cast<data_t>(r * 100 + c)) { valsOK = false; break; }

        logExpect(valsOK, failures, log, "matrix2D indexing preserves written values");
    }

    // ---------- Test 1: matrix3D size, indexing & contiguity ----------
    {
        constexpr std::size_t D = 2, R = 3, C = 4;
        matrix3D<D, R, C> m3{};

        // size check
        bool sizeOK = sizeof(m3) == D * R * C * sizeof(data_t);
        logExpect(sizeOK, failures, log, "matrix3D sizeof == D*R*C*sizeof(data_t)");

        // fill linearly so we can check contiguity and indexing
        data_t v = 0;
        for (std::size_t d = 0; d < D; ++d)
            for (std::size_t r = 0; r < R; ++r)
                for (std::size_t c = 0; c < C; ++c)
                    m3[d][r][c] = v++;

        // indexing check
        bool idxOK = true;
        v = 0;
        for (std::size_t d = 0; d < D && idxOK; ++d)
            for (std::size_t r = 0; r < R && idxOK; ++r)
                for (std::size_t c = 0; c < C; ++c)
                    if (m3[d][r][c] != v++) { idxOK = false; break; }
        logExpect(idxOK, failures, log, "matrix3D indexing returns expected values");

        // contiguity check: row-major contiguous memory
        const data_t* base = &m3[0][0][0];
        bool contigOK = true;
        for (std::size_t i = 0; i < D * R * C; ++i) {
            if (base[i] != static_cast<data_t>(i)) { contigOK = false; break; }
        }
        logExpect(contigOK, failures, log, "matrix3D memory is contiguous (row-major)");
    }

    // ---------- Test 2: matrix4D size, indexing & contiguity ----------
    {
        constexpr std::size_t B = 2, D = 2, R = 2, C = 3;
        matrix4D<B, D, R, C> m4{};

        // size check
        bool sizeOK = sizeof(m4) == B * D * R * C * sizeof(data_t);
        logExpect(sizeOK, failures, log, "matrix4D sizeof == B*D*R*C*sizeof(data_t)");

        // fill linearly
        data_t v = 0;
        for (std::size_t b = 0; b < B; ++b)
            for (std::size_t d = 0; d < D; ++d)
                for (std::size_t r = 0; r < R; ++r)
                    for (std::size_t c = 0; c < C; ++c)
                        m4[b][d][r][c] = v++;

        // indexing check
        bool idxOK = true;
        v = 0;
        for (std::size_t b = 0; b < B && idxOK; ++b)
            for (std::size_t d = 0; d < D && idxOK; ++d)
                for (std::size_t r = 0; r < R && idxOK; ++r)
                    for (std::size_t c = 0; c < C; ++c)
                        if (m4[b][d][r][c] != v++) { idxOK = false; break; }
        logExpect(idxOK, failures, log, "matrix4D indexing returns expected values");

        // contiguity check
        const data_t* base = &m4[0][0][0][0];
        bool contigOK = true;
        for (std::size_t i = 0; i < B * D * R * C; ++i) {
            if (base[i] != static_cast<data_t>(i)) { contigOK = false; break; }
        }
        logExpect(contigOK, failures, log, "matrix4D memory is contiguous (row-major)");
    }

    // ---------- Summary ----------
    if (failures == 0) log << "All CNN alias tests PASSED.\n";
    else               log << failures << " CNN alias test(s) FAILED.\n";

    log.close();
    return failures;
}
