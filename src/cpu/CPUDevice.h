#pragma 
#include "../common/Device.h"


class CPUDevice : public Device {
public:
    virtual ~CPUDevice() {}

    double dot(const std::vector<double>& a, const std::vector<double>& b) override;

    std::vector<double> matmul(const std::vector<double>& A, const std::vector<double>& B,
        int A_rows, int A_cols, int B_cols) override;

    void applyActivation(std::vector<double>& data, double (*activation)(double)) override;

    void cleanup() override {};
};