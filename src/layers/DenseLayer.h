#pragma once

#include "Layer.h"
#include <random>
#include <iostream>

// Fully connected (dense) layer
class DenseLayer : public Layer {
private:
    int inputDim;
    int outputDim;
    Device* device;
    
    // Layer parameters
    Eigen::MatrixXd weights;     // (outputDim x inputDim)
    Eigen::VectorXd biases;      // (outputDim)
    
    // Adam optimizer accumulators
    Eigen::MatrixXd m_w, v_w;    // For weights
    Eigen::VectorXd m_b, v_b;    // For biases
    
    // Adam hyperparameters
    double beta1 = 0.9;
    double beta2 = 0.999;
    double epsilon = 1e-8;
    int t = 0;  // Timestep counter
    
    // Gradient clipping threshold
    double clipThreshold = 5.0;
    
    // Cached data for backprop
    Eigen::MatrixXd lastInput;
    Eigen::MatrixXd lastOutput;
    
    // Activation function
    Activation activation;
    ActivationType actType;
    
    // Weight decay (L2 regularization)
    double weightDecay = 0.0001;

public:
    DenseLayer(int outputDim, ActivationType actType = RELU, double weightDecay = 0.0001)
        : outputDim(outputDim), actType(actType), activation(getActivation(actType)), 
          weightDecay(weightDecay) {}
    
    void initialize(int inputDim, int outputDim, Device* device) override {
        this->inputDim = inputDim;
        this->outputDim = outputDim;
        this->device = device;
        
        // Initialize weights using He initialization for ReLU (or variants)
        double scale;
        if (actType == RELU || actType == LEAKY_RELU || actType == ELU) {
            // He initialization: scale = sqrt(2 / fan_in)
            scale = std::sqrt(2.0 / inputDim);
        } else {
            // Xavier/Glorot initialization: scale = sqrt(6 / (fan_in + fan_out))
            scale = std::sqrt(6.0 / (inputDim + outputDim));
        }
        
        // Create random number generator
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<double> dist(0.0, 1.0);
        
        // Initialize weights, biases, and Adam accumulators
        weights = Eigen::MatrixXd::Zero(outputDim, inputDim);
        biases = Eigen::VectorXd::Zero(outputDim);
        
        // Fill weights with random values
        for (int i = 0; i < outputDim; i++) {
            for (int j = 0; j < inputDim; j++) {
                weights(i, j) = dist(gen) * scale;
            }
        }
        
        // Initialize optimizer accumulators
        m_w = Eigen::MatrixXd::Zero(outputDim, inputDim);
        v_w = Eigen::MatrixXd::Zero(outputDim, inputDim);
        m_b = Eigen::VectorXd::Zero(outputDim);
        v_b = Eigen::VectorXd::Zero(outputDim);
    }
    
    Eigen::MatrixXd forward(const Eigen::MatrixXd& input, bool isTraining) override {
        lastInput = input; // Cache for backprop
        
        // Compute Z = X * W^T + b
        Eigen::MatrixXd Z;
        if (dynamic_cast<MetalDevice*>(device)) {
            // Use GPU acceleration if available
            MetalDevice* gpu = dynamic_cast<MetalDevice*>(device);
            Z = gpu->matmulGPU(input, weights, biases);
        } else {
            // CPU fallback
            Z = input * weights.transpose();
            Z.rowwise() += biases.transpose();
        }
        
        // Apply activation function
        if (actType != SOFTMAX) {
            // Element-wise activation for non-softmax
            lastOutput = Z.unaryExpr([this](double z) { return activation.func(z); });
            return lastOutput;
        } else {
            // Special handling for softmax
            lastOutput = Eigen::MatrixXd(Z.rows(), Z.cols());
            
            for (int i = 0; i < Z.rows(); i++) {
                // Subtract max for numerical stability
                double maxVal = Z.row(i).maxCoeff();
                Eigen::VectorXd exps = (Z.row(i).array() - maxVal).exp();
                double sumExps = exps.sum();
                lastOutput.row(i) = exps.transpose() / sumExps;
            }
            
            return lastOutput;
        }
    }
    
    Eigen::MatrixXd backward(const Eigen::MatrixXd& gradOutput, double learningRate) override {
        t++; // Increment timestep for Adam
        int batchSize = lastInput.rows();
        
        // Compute gradient through the activation function
        Eigen::MatrixXd delta;
        if (actType != SOFTMAX) {
            // For non-softmax, multiply by the derivative
            Eigen::MatrixXd actDeriv = lastOutput.unaryExpr(
                [this](double y) { return activation.derivative(y); }
            );
            delta = gradOutput.cwiseProduct(actDeriv);
        } else {
            // For softmax with cross-entropy loss, the gradient is already correct
            delta = gradOutput;
        }
        
        // Compute gradient with respect to weights: dW = delta^T * X
        Eigen::MatrixXd dW = delta.transpose() * lastInput / batchSize;
        
        // Add L2 regularization gradient
        dW += weightDecay * weights;
        
        // Compute gradient with respect to biases: db = sum(delta, axis=0)
        Eigen::VectorXd db = delta.colwise().sum() / batchSize;
        
        // Apply gradient clipping
        double normDW = dW.norm();
        if (normDW > clipThreshold) {
            dW *= (clipThreshold / normDW);
        }
        
        double normDB = db.norm();
        if (normDB > clipThreshold) {
            db *= (clipThreshold / normDB);
        }
        
        // Update parameters using Adam optimizer
        
        // Update first moment for weights
        m_w = beta1 * m_w + (1 - beta1) * dW;
        
        // Update second moment for weights
        v_w = beta2 * v_w + (1 - beta2) * dW.cwiseProduct(dW);
        
        // Update first moment for biases
        m_b = beta1 * m_b + (1 - beta1) * db;
        
        // Update second moment for biases
        v_b = beta2 * v_b + (1 - beta2) * db.cwiseProduct(db);
        
        // Bias correction
        double bias_correction1 = 1.0 - std::pow(beta1, t);
        double bias_correction2 = 1.0 - std::pow(beta2, t);
        
        // Update weights - FIX: Use array operations instead of cwiseQuotient
        Eigen::MatrixXd m_w_hat = m_w / bias_correction1;
        Eigen::MatrixXd v_w_hat = v_w / bias_correction2;
        Eigen::MatrixXd w_update = learningRate * (m_w_hat.array() / 
            (v_w_hat.array().sqrt() + epsilon)).matrix();
        weights -= w_update;
        
        // Update biases - FIX: Use array operations instead of cwiseQuotient
        Eigen::VectorXd m_b_hat = m_b / bias_correction1;
        Eigen::VectorXd v_b_hat = v_b / bias_correction2;
        Eigen::VectorXd b_update = learningRate * (m_b_hat.array() / 
            (v_b_hat.array().sqrt() + epsilon)).matrix();
        biases -= b_update;
        
        // Compute gradient with respect to input: dX = delta * W
        return delta * weights;
    }
    
    int getOutputSize() const override {
        return outputDim;
    }
    
    std::string getName() const override {
        return "DenseLayer";
    }
    
    void saveParameters(std::ostream& os) const override {
        // Save weights and biases
        for (int i = 0; i < outputDim; i++) {
            for (int j = 0; j < inputDim; j++) {
                os << weights(i, j) << " ";
            }
            os << "\n";
        }
        
        for (int i = 0; i < outputDim; i++) {
            os << biases(i) << " ";
        }
        os << "\n";
    }
    
    void loadParameters(std::istream& is) override {
        // Load weights and biases
        for (int i = 0; i < outputDim; i++) {
            for (int j = 0; j < inputDim; j++) {
                is >> weights(i, j);
            }
        }
        
        for (int i = 0; i < outputDim; i++) {
            is >> biases(i);
        }
    }
};