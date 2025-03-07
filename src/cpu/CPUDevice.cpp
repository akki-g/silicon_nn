#include "CPUDevice.h"
#include <Eigen/Dense>


double CPUDevice::dot(const std::vector<double>& a, const std::vector<double>& b) {
    Eigen::Map<const Eigen::VectorXd> A(a.data(), a.size());
    Eigen::Map<const Eigen::VectorXd> B(b.data(), b.size());

    return A.dot(B);
}


std::vector<double> CPUDevice::matmul(const std::vector<double>& A, const std::vector<double>& B, int A_rows, int A_cols, int B_cols) {

    Eigen::Map<const Eigen::MatrixXd> a(A.data(), A_rows, A_cols);
    Eigen::Map<const Eigen::MatrixXd> b(B.data(), A_cols, B_cols);

    Eigen::MatrixXd c = a * b;

    std::vector<double> result(c.size());
    Eigen::VectorXd::Map(&result[0], c.size()) = c;

    return result;
}


void CPUDevice::applyActivation(std::vector<double>& data, double (*activation)(double)) {
    for (auto& d : data) {
        d = activation(d);
    }
}