#pragma once
#include <vector>
#include <iostream>

// Define the available device types. For now, CPU and GPU (via Metal on macOS).
enum class DeviceType {
    CPU,
    GPU // GPU will use Metal backend on macOS
};

// Device is an abstract base class (interface) that declares the common operations
// required for neural network computations.
class Device {
public:
    virtual ~Device() {}

    // Compute the dot product between two vectors.
    // Each vector is assumed to have the same length.
    virtual double dot(const std::vector<double>& a, const std::vector<double>& b) = 0;

    // Perform matrix multiplication.
    // A is an A_rows x A_cols matrix and B is an A_cols x B_cols matrix.
    // The result is returned as a vector in row-major order of size (A_rows * B_cols).
    virtual std::vector<double> matmul(const std::vector<double>& A, const std::vector<double>& B,
                                         int A_rows, int A_cols, int B_cols) = 0;

    // Apply an activation function element-wise to the data vector.
    // The activation is provided as a pointer to a function.
    virtual void applyActivation(std::vector<double>& data, double (*activation)(double)) = 0;

    // Factory method: Given a device type, returns an instance of the appropriate Device.
    static Device* create(DeviceType type);
};

//
// Inline implementation of the factory method.
// This includes the CPU and Metal device headers so that we can instantiate them.
//
#include "CPUDevice.h"     // Your CPU-specific implementation header.
#include "MetalDevice.h"   // Your Metal-specific implementation header.

// Factory method implementation.
// It creates and returns a pointer to a Device (CPUDevice or MetalDevice)
// based on the provided DeviceType.
static inline Device* Device::create(DeviceType type) {
    switch (type) {
        case DeviceType::CPU:
            return new CPUDevice();
        case DeviceType::GPU:
            return new MetalDevice();
        default:
            std::cerr << "Unknown device type!" << std::endl;
            return nullptr;
    }
}
