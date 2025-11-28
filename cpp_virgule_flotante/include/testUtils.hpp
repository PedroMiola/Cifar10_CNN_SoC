#ifndef TESTUTILS_HPP
#define TESTUTILS_HPP

#include <fstream>
#include <string>

void logExpect(bool cond, int& fails, std::ofstream& log, const std::string& msg) {
    if (cond) { log << "[ OK ] " << msg << std::endl; }
    else { log << "[FAIL] " << msg << std::endl; ++fails; }
}

#endif // TESTUTILS_HPP