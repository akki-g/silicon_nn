#pragma once

#ifdef __OBJC__
    // When compiling as Objective-C++ (e.g. in .mm files), import the Metal framework.
    #import <Metal/Metal.h>
#else
    // When not compiling as Objective-C++, define dummy types.
    typedef void* MTLDeviceRef;
    typedef void* MTLCommandQueueRef;
    // Optionally, you can also define 'id' if needed:
    typedef void* id;
#endif

#include "../common/Device.h"
#include <vector>

// MetalDevice is a subclass of Device that uses Apple's Metal for GPU acceleration.
class MetalDevice : public Device {
public:
    MetalDevice();
    virtual ~MetalDevice();

    // Overrides of Device methods
    double dot(const std::vector<double>& a, const std::vector<double>& b) override;
    std::vector<double> matmul(const std::vector<double>& A, const std::vector<double>& B,
                               int A_rows, int A_cols, int B_cols) override;
    void applyActivation(std::vector<double>& data, double (*activation)(double)) override;
    
    void cleanup() override;
    // Static method to check if Metal is available.
    static bool metalIsAvailable();


private:
#ifdef __OBJC__
    id<MTLDevice> metalDevice;
    id<MTLCommandQueue> commandQueue;
#else
    MTLDeviceRef metalDevice;
    MTLCommandQueueRef commandQueue;
#endif
    bool isCleaned = false;
};
