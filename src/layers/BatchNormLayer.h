#pragma once

#include "Layer.h"

// Batch Normalization Layer
class BatchNormLayer : public Layer {
private:
    int dimension;
    Device* device;
    
    // Learnable parameters
    Eigen::VectorXd gamma;      // Scale parameter (dimension)
    Eigen::VectorXd beta;       // Shift parameter (dimension)
    
    // Running statistics for inference
    Eigen::VectorXd runningMean;
    Eigen::VectorXd runningVar;
    
    // Adam optimizer accumulators
    Eigen::VectorXd m_gamma, v_gamma;
    Eigen::VectorXd m_beta, v_beta;
    
    // Adam hyperparameters
    double beta1 = 0.9;
    double beta2 = 0.999;
    double epsilon = 1e-8;
    int t = 0;  // Timestep counter
    
    // Batch norm hyperparameters
    double momentum = 0.9;
    double eps = 1e-5;  // Small constant for numerical stability
    
    // Cached values for backprop
    Eigen::MatrixXd xNorm;
    Eigen::MatrixXd xCentered;
    Eigen::VectorXd batchVar;
    Eigen::VectorXd batchMean;
    Eigen::VectorXd invStd;

public:
    BatchNormLayer(double momentum = 0.9)
        : momentum(momentum) {}
    
    void initialize(int inputDim, int outputDim, Device* device) override {
        this->dimension = inputDim;
        this->device = device;
        
        // Initialize learnable parameters
        gamma = Eigen::VectorXd::Ones(dimension);
        beta = Eigen::VectorXd::Zero(dimension);
        
        // Initialize running statistics
        runningMean = Eigen::VectorXd::Zero(dimension);
        runningVar = Eigen::VectorXd::Ones(dimension);
        
        // Initialize Adam accumulators
        m_gamma = Eigen::VectorXd::Zero(dimension);
        v_gamma = Eigen::VectorXd::Zero(dimension);
        m_beta = Eigen::VectorXd::Zero(dimension);
        v_beta = Eigen::VectorXd::Zero(dimension);
    }
    
    Eigen::MatrixXd forward(const Eigen::MatrixXd& input, bool isTraining) override {
        if (isTraining) {
            // Training mode: use batch statistics
            int batchSize = input.rows();
            
            // Compute batch mean
            batchMean = input.colwise().mean();
            
            // Center the data (x - mean)
            xCentered = input.rowwise() - batchMean.transpose();
            
            // Compute batch variance
            batchVar = (xCentered.array().square().colwise().sum() / batchSize).matrix();
            
            // Compute inverse standard deviation (1 / sqrt(var + eps))
            invStd = (batchVar.array() + eps).sqrt().inverse().matrix();
            
            // Normalize the data
            xNorm = xCentered.array().rowwise() * invStd.transpose().array();
            
            // Scale and shift
            Eigen::MatrixXd out = xNorm.array().rowwise() * gamma.transpose().array();
            out = out.array().rowwise() + beta.transpose().array();
            
            // Update running statistics for inference
            runningMean = momentum * runningMean + (1 - momentum) * batchMean;
            runningVar = momentum * runningVar + (1 - momentum) * batchVar;
            
            return out;
        } else {
            // Inference mode: use running statistics
            Eigen::MatrixXd xCentered = input.rowwise() - runningMean.transpose();
            Eigen::MatrixXd xNorm = xCentered.array().rowwise() * 
                                    (runningVar.array() + eps).sqrt().inverse().matrix().transpose().array();
            
            // Scale and shift
            return xNorm.array().rowwise() * gamma.transpose().array() + 
                   beta.transpose().array();
        }
    }
    
    Eigen::MatrixXd backward(const Eigen::MatrixXd& gradOutput, double learningRate) override {
        t++;  // Increment timestep for Adam
        int batchSize = gradOutput.rows();
        
        // Gradient with respect to scale parameter (gamma)
        Eigen::VectorXd dGamma = (gradOutput.cwiseProduct(xNorm)).colwise().sum();
        
        // Gradient with respect to shift parameter (beta)
        Eigen::VectorXd dBeta = gradOutput.colwise().sum();
        
        // Gradient with respect to normalized input (xNorm)
        Eigen::MatrixXd dxNorm = gradOutput.array().rowwise() * gamma.transpose().array();
        
        // Gradient with respect to batch variance
        Eigen::VectorXd dVar = (dxNorm.cwiseProduct(xCentered) * -0.5).colwise().sum();
        dVar = dVar.cwiseProduct(invStd.cwiseProduct(invStd).cwiseProduct(invStd));
        
        // Gradient with respect to batch mean
        Eigen::VectorXd dMean = (dxNorm.array().rowwise() * (-invStd.transpose().array())).colwise().sum();
        dMean -= 2.0 * dVar.cwiseProduct(xCentered.colwise().sum()) / batchSize;
        
        // Gradient with respect to input
        Eigen::MatrixXd dx = dxNorm.array().rowwise() * invStd.transpose().array();
        dx += (dVar * 2.0 / batchSize).transpose().replicate(batchSize, 1).cwiseProduct(xCentered);
        dx.rowwise() += (dMean / batchSize).transpose();
        
        // Update parameters using Adam optimizer
        
        // Update first moment for gamma
        m_gamma = beta1 * m_gamma + (1 - beta1) * dGamma;
        
        // Update second moment for gamma
        v_gamma = beta2 * v_gamma + (1 - beta2) * dGamma.cwiseProduct(dGamma);
        
        // Update first moment for beta
        m_beta = beta1 * m_beta + (1 - beta1) * dBeta;
        
        // Update second moment for beta
        v_beta = beta2 * v_beta + (1 - beta2) * dBeta.cwiseProduct(dBeta);
        
        // Bias correction
        double bias_correction1 = 1.0 - std::pow(beta1, t);
        double bias_correction2 = 1.0 - std::pow(beta2, t);
        
        // Update gamma - FIX: Use array operations instead of cwiseQuotient
        Eigen::VectorXd m_gamma_hat = m_gamma / bias_correction1;
        Eigen::VectorXd v_gamma_hat = v_gamma / bias_correction2;
        Eigen::VectorXd gamma_update = learningRate * (m_gamma_hat.array() / 
            (v_gamma_hat.array().sqrt() + epsilon)).matrix();
        gamma -= gamma_update;
        
        // Update beta - FIX: Use array operations instead of cwiseQuotient
        Eigen::VectorXd m_beta_hat = m_beta / bias_correction1;
        Eigen::VectorXd v_beta_hat = v_beta / bias_correction2;
        Eigen::VectorXd beta_update = learningRate * (m_beta_hat.array() / 
            (v_beta_hat.array().sqrt() + epsilon)).matrix();
        beta -= beta_update;
        
        return dx;
    }
    
    int getOutputSize() const override {
        return dimension;
    }
    
    std::string getName() const override {
        return "BatchNormLayer";
    }
    
    void saveParameters(std::ostream& os) const override {
        // Save gamma and beta parameters
        for (int i = 0; i < dimension; i++) {
            os << gamma(i) << " ";
        }
        os << "\n";
        
        for (int i = 0; i < dimension; i++) {
            os << beta(i) << " ";
        }
        os << "\n";
        
        // Save running mean and variance
        for (int i = 0; i < dimension; i++) {
            os << runningMean(i) << " ";
        }
        os << "\n";
        
        for (int i = 0; i < dimension; i++) {
            os << runningVar(i) << " ";
        }
        os << "\n";
    }
    
    void loadParameters(std::istream& is) override {
        // Load gamma and beta parameters
        for (int i = 0; i < dimension; i++) {
            is >> gamma(i);
        }
        
        for (int i = 0; i < dimension; i++) {
            is >> beta(i);
        }
        
        // Load running mean and variance
        for (int i = 0; i < dimension; i++) {
            is >> runningMean(i);
        }
        
        for (int i = 0; i < dimension; i++) {
            is >> runningVar(i);
        }
    }
};