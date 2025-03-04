#pragma once
#include "../common/Device.h"
#include <vector>

struct MTLDevice;
struct MTLCommandQueue;

class MetalDevice : public Device {
    public: 
        MetalDevice() {}
        virtual ~MetalDevice() {}

        double dot(const std::vector<double>& a, const std::vector<double>& b) override;
        std::vector<double> matmul(const std::vector<double>& A, const std::vector<double>& B,
                                    int A_rows, int A_cols, int B_cols) override;
        void applyActivation(std::vector<double>& data, double (*activation)(double)) override;

        static bool metalIsAvailable();
    private:
        // Metal-specific implementation details
        // ...
        MTLDevice* metalDevice;
        MTLCommandQueue* commandQueue;
};