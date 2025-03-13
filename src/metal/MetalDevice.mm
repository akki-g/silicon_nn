#import "MetalDevice.h"
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>
#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;

MetalDevice::MetalDevice() {
    metalDevice = MTLCreateSystemDefaultDevice();
    if (metalDevice) {
        commandQueue = [metalDevice newCommandQueue];
        std::cout << "MetalDevice created" << std::endl;
    } else {
        std::cerr << "Metal device not found" << std::endl;
        exit(1);
    }
}

MetalDevice::~MetalDevice() {
    if (!isCleaned) {
        cleanup();
    }
    std::cout << "MetalDevice: Destructor called" << std::endl;
}

double MetalDevice::dot(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) {
        std::cerr << "Vectors must have the same size" << std::endl;
        exit(1);
    }
    double result = 0;
    for (int i = 0; i < a.size(); i++) {
        result += a[i] * b[i];
    }
    return result;
}

std::vector<double> MetalDevice::matmul(const std::vector<double>& A, const std::vector<double>& B,
                                        int A_rows, int A_cols, int B_cols) {
    if (A.size() != A_rows * A_cols || B.size() != A_cols * B_cols) {
        std::cerr << "Matrix dimensions do not match" << std::endl;
        exit(1);
    }
    std::vector<double> C(A_rows * B_cols, 0);
    for (int i = 0; i < A_rows; i++) {
        for (int j = 0; j < B_cols; j++) {
            for (int k = 0; k < A_cols; k++) {
                C[i * B_cols + j] += A[i * A_cols + k] * B[k * B_cols + j];
            }
        }
    }
    return C;
}

void MetalDevice::applyActivation(std::vector<double>& A, double (*activation)(double)) {
    for (int i = 0; i < A.size(); i++) {
        A[i] = activation(A[i]);
    }
}

// Stub for GPU-accelerated matrix multiply using Metal.
// For now, we simply fallback to CPU using Eigen.
MatrixXd MetalDevice::matmulGPU(const MatrixXd &A, const MatrixXd &W, const VectorXd &b) {
    MatrixXd Z = A * W.transpose();
    Z.rowwise() += b.transpose();
    return Z;
}

bool MetalDevice::metalIsAvailable() {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    return (dev != nil);
}

void MetalDevice::cleanup() {
    if (isCleaned) return;
    std::cout << "MetalDevice: Cleanup called, releasing Metal resources." << std::endl;
#ifdef __OBJC__
    if (commandQueue) {
        commandQueue = nil;
    }
    if (metalDevice) {
        metalDevice = nil;
    }
#endif
    isCleaned = true;
}
