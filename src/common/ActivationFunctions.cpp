#include "ActivationFunctions.h"
#include <cmath>

// Basic activation functions
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoidDerivative(double y) {
    return y * (1.0 - y);
}

double relu(double x) {
    return x > 0 ? x : 0.0;
}

double reluDerivative(double y) {
    return y > 0 ? 1.0 : 0.0;
}

double tanhActivation(double x) {
    return tanh(x);
}

double tanhDerivative(double y) {
    return 1.0 - y * y;
}

// Advanced activation functions
double leakyRelu(double x, double alpha) {
    return x > 0 ? x : alpha * x;
}

double leakyReluDerivative(double y, double alpha) {
    return y > 0 ? 1.0 : alpha;
}

double elu(double x, double alpha) {
    return x > 0 ? x : alpha * (exp(x) - 1.0);
}

double eluDerivative(double y, double alpha) {
    return y > 0 ? 1.0 : y + alpha;
}

// Wrapper functions with simple double(double) signature
double leakyReluWrapper(double x) {
    return leakyRelu(x, 0.01); // Use default alpha of 0.01
}

double eluWrapper(double x) {
    return elu(x, 1.0); // Use default alpha of 1.0
}

// For softmax, we handle it in the forward pass
double softmax(double x) {
    return exp(x);
}

double softmaxDerivative(double y) {
    return 1.0;
}