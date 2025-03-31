#pragma once

#include <Eigen/Dense>
#include "../common/Device.h"

enum ActivationType { SIGMOID = 0, RELU = 1, TANH = 2, SOFTMAX = 3, LEAKY_RELU = 4, ELU = 5};

struct Activation {
    std::function<double(double)> func;
    std::function<double(double)> derivative;
};

inline double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}
inline double sigmoidDerivative(double y) {
    return y * (1.0 - y);
}

inline double relu(double x) {
    return x > 0 ? x : 0.0;
}
inline double reluDerivative(double y) {
    return y > 0 ? 1.0 : 0.0;
}

inline double leakyRelu(double x, double alpha = 0.01) {
    return x > 0 ? x : alpha * x;
}
inline double leakyReluDerivative(double y, double alpha = 0.01) {
    return y > 0 ? 1.0 : alpha;
}

inline double tanhActivation(double x) {
    return tanh(x);
}
inline double tanhDerivative(double y) {
    return 1.0 - y * y;
}

inline double elu(double x, double alpha = 1.0) {
    return x > 0 ? x : alpha * (exp(x) - 1);
}
inline double eluDerivative(double y, double alpha = 1.0) {
    return y > 0 ? 1.0 : y + alpha;
}


inline Activation getActivation(ActivationType type) {
    switch (type) {
        case SIGMOID: return { sigmoid, sigmoidDerivative };
        case RELU:    return { relu, reluDerivative };
        case TANH:    return { tanhActivation, tanhDerivative };
        case SOFTMAX: return { [](double x){ return x; }, [](double y){ return 1.0; } };
        case LEAKY_RELU:
            return { 
                [](double x) { return leakyRelu(x); }, 
                [](double y) { return leakyReluDerivative(y); } 
            };
        case ELU:
            return { 
                [](double x) { return elu(x); }, 
                [](double y) { return eluDerivative(y); } 
            };
        default:
            std::cerr << "Invalid activation type." << std::endl;
            exit(1);
    }
}


class Layer {
public: 
    virtual ~Layer() {}

    virtual Eigen::MatrixXd forward(const Eigen::MatrixXd& input, bool isTraining) = 0;

    virtual Eigen::MatrixXd backward(const Eigen::MatrixXd& gradOutput, double learningRate) = 0;

    virtual void initialize(int inputDim, int outputDim, Device* device) = 0;

    virtual int getOutputSize() const = 0;

    virtual std::string getName() const = 0;

    virtual void saveParameters(std::ostream& os) const {

    }

    virtual void loadParameters(std::istream& is) {

    }
};