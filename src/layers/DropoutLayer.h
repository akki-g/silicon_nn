#pragma once

#include "Layer.h"
#include <random>

// Dropout layer
class DropoutLayer : public Layer {
private:
    int dimension;
    double dropoutRate;
    Device* device;
    
    // Dropout mask (1 for keep, 0 for drop)
    Eigen::MatrixXd mask;
    
    // Random number generator
    std::mt19937 gen;

public:
    DropoutLayer(double dropoutRate = 0.5)
        : dropoutRate(dropoutRate), gen(std::random_device()()) {}
    
    void initialize(int inputDim, int outputDim, Device* device) override {
        this->dimension = inputDim;
        this->device = device;
    }
    
    Eigen::MatrixXd forward(const Eigen::MatrixXd& input, bool isTraining) override {
        if (!isTraining || dropoutRate <= 0.0) {
            // During testing, no dropout is applied
            return input;
        }
        
        // Create new mask for this forward pass
        int batchSize = input.rows();
        mask = Eigen::MatrixXd::Ones(batchSize, dimension);
        
        // Generate random mask
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < dimension; j++) {
                if (dist(gen) < dropoutRate) {
                    mask(i, j) = 0.0;  // Drop this activation
                }
            }
        }
        
        // Apply mask and scale
        double scale = 1.0 / (1.0 - dropoutRate);  // Inverted dropout
        return input.cwiseProduct(mask) * scale;
    }
    
    Eigen::MatrixXd backward(const Eigen::MatrixXd& gradOutput, double learningRate) override {
        // Apply the same mask to gradients
        // Note: we don't need to rescale here since we already scaled in the forward pass
        return gradOutput.cwiseProduct(mask);
    }
    
    int getOutputSize() const override {
        return dimension;
    }
    
    std::string getName() const override {
        return "DropoutLayer";
    }
};