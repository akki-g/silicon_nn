#pragma once

// Basic activation functions
double sigmoid(double x);
double sigmoidDerivative(double y);

double relu(double x);
double reluDerivative(double y);

double tanhActivation(double x);
double tanhDerivative(double y);

// Advanced activation functions with parameters
double leakyRelu(double x, double alpha = 0.01);
double leakyReluDerivative(double y, double alpha = 0.01);

double elu(double x, double alpha = 1.0);
double eluDerivative(double y, double alpha = 1.0);

// Wrapper functions with simple double(double) signature for compatibility
// These use the default parameter values
double leakyReluWrapper(double x);
double eluWrapper(double x);

// For softmax, we handle it differently
double softmax(double x);
double softmaxDerivative(double y);