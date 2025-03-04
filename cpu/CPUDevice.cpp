#include "CPUDevice.h"
#include <cassert>

double CPUDevice::dot(const std::vector<double>& a, const std::vector<double>& b) {
    assert(a.size() == b.size());
    double result = 0;
    for (size_t i = 0; i < a.size(); i++) {
        result += a[i] * b[i];
    }
    return result;
}


std::vector<double> CPUDevice::matmul(const std::vector<double>& A, const std::vector<double>& B, int A_rows, int A_cols, int B_cols) {

    std::vector<double> result(A_rows * B_cols, 0.0);
    for (int i = 0; i < A_rows; ++i) {
        for (int j = 0; j < B_cols; ++j) {
            for (int k = 0; k < A_cols; ++k) {
                result[i * B_cols + j] += A[i * A_cols + k] * B[k * B_cols + j];
            }
        }
    }
    return result;
}


void CPUDevice::applyActivation(std::vector<double>& data, double (*activation)(double)) {
    for (auto& d : data) {
        d = activation(d);
    }
}