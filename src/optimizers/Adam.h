#pragma once

#include <vector>
#include <cmath>
#include <Eigen/Dense>

// Abstract Optimizer interface
class Optimizer {
public:
    virtual void updateVector(std::vector<double>& param,
                              std::vector<double>& m,
                              std::vector<double>& v,
                              const std::vector<double>& grad) = 0;
    
    virtual void updateScalar(double &param,
                              double &m,
                              double &v,
                              double grad) = 0;
    
    virtual void updateMatrix(Eigen::MatrixXd& param,
                              Eigen::MatrixXd& m,
                              Eigen::MatrixXd& v,
                              const Eigen::MatrixXd& grad) = 0;
                              
    virtual void updateVector(Eigen::VectorXd& param,
                              Eigen::VectorXd& m,
                              Eigen::VectorXd& v,
                              const Eigen::VectorXd& grad) = 0;
    
    virtual ~Optimizer() {}
};

// AdamW optimizer with decoupled weight decay
class AdamW : public Optimizer {
private:
    double learningRate;
    double beta1;
    double beta2;
    double epsilon;
    double weightDecay;
    int t;  // Current timestep
    
public:
    AdamW(double lr = 0.001, double wd = 0.0001, double b1 = 0.9, double b2 = 0.999, double eps = 1e-8)
      : learningRate(lr), weightDecay(wd), beta1(b1), beta2(b2), epsilon(eps), t(0) {}
    
    // Update a vector parameter (using std::vector)
    void updateVector(std::vector<double>& param,
                      std::vector<double>& m,
                      std::vector<double>& v,
                      const std::vector<double>& grad) override {
        t++;
        double bias_correction1 = 1.0 - std::pow(beta1, t);
        double bias_correction2 = 1.0 - std::pow(beta2, t);
        
        for (size_t i = 0; i < param.size(); i++) {
            // Update first moment
            m[i] = beta1 * m[i] + (1.0 - beta1) * grad[i];
            
            // Update second moment
            v[i] = beta2 * v[i] + (1.0 - beta2) * grad[i] * grad[i];
            
            // Bias correction
            double m_hat = m[i] / bias_correction1;
            double v_hat = v[i] / bias_correction2;
            
            // Update parameter
            param[i] -= learningRate * (m_hat / (std::sqrt(v_hat) + epsilon) + weightDecay * param[i]);
        }
    }
    
    // Update a scalar parameter
    void updateScalar(double &param,
                      double &m,
                      double &v,
                      double grad) override {
        t++;
        double bias_correction1 = 1.0 - std::pow(beta1, t);
        double bias_correction2 = 1.0 - std::pow(beta2, t);
        
        // Update first moment
        m = beta1 * m + (1.0 - beta1) * grad;
        
        // Update second moment
        v = beta2 * v + (1.0 - beta2) * grad * grad;
        
        // Bias correction
        double m_hat = m / bias_correction1;
        double v_hat = v / bias_correction2;
        
        // Update parameter (typically no weight decay for bias)
        param -= learningRate * m_hat / (std::sqrt(v_hat) + epsilon);
    }
    
    // Update a matrix parameter (using Eigen::MatrixXd)
    void updateMatrix(Eigen::MatrixXd& param,
                      Eigen::MatrixXd& m,
                      Eigen::MatrixXd& v,
                      const Eigen::MatrixXd& grad) override {
        t++;
        double bias_correction1 = 1.0 - std::pow(beta1, t);
        double bias_correction2 = 1.0 - std::pow(beta2, t);
        
        // Update first moment
        m = beta1 * m + (1.0 - beta1) * grad;
        
        // Update second moment
        v = beta2 * v + (1.0 - beta2) * grad.cwiseProduct(grad);
        
        // Compute update - FIX: Use array operations instead of cwiseQuotient
        Eigen::MatrixXd m_hat = m / bias_correction1;
        Eigen::MatrixXd v_hat = v / bias_correction2;
        Eigen::MatrixXd update = (m_hat.array() / (v_hat.array().sqrt() + epsilon)).matrix();
        
        // Apply weight decay
        param -= learningRate * (update + weightDecay * param);
    }
    
    // Update a vector parameter (using Eigen::VectorXd)
    void updateVector(Eigen::VectorXd& param,
                      Eigen::VectorXd& m,
                      Eigen::VectorXd& v,
                      const Eigen::VectorXd& grad) override {
        t++;
        double bias_correction1 = 1.0 - std::pow(beta1, t);
        double bias_correction2 = 1.0 - std::pow(beta2, t);
        
        // Update first moment
        m = beta1 * m + (1.0 - beta1) * grad;
        
        // Update second moment
        v = beta2 * v + (1.0 - beta2) * grad.cwiseProduct(grad);
        
        // Compute update - FIX: Use array operations instead of cwiseQuotient
        Eigen::VectorXd m_hat = m / bias_correction1;
        Eigen::VectorXd v_hat = v / bias_correction2;
        Eigen::VectorXd update = (m_hat.array() / (v_hat.array().sqrt() + epsilon)).matrix();
        
        // Apply update (typically no weight decay for bias)
        param -= learningRate * update;
    }
};

// SGD optimizer with momentum and Nesterov acceleration
class SGD : public Optimizer {
private:
    double learningRate;
    double momentum;
    double weightDecay;
    bool nesterov;
    
public:
    SGD(double lr = 0.01, double mom = 0.9, double wd = 0.0005, bool nesterov = true)
      : learningRate(lr), momentum(mom), weightDecay(wd), nesterov(nesterov) {}
    
    // Update a vector parameter (using std::vector)
    void updateVector(std::vector<double>& param,
                      std::vector<double>& velocity,
                      std::vector<double>& unused,
                      const std::vector<double>& grad) override {
        for (size_t i = 0; i < param.size(); i++) {
            // Apply weight decay
            double grad_with_decay = grad[i] + weightDecay * param[i];
            
            // Update velocity
            velocity[i] = momentum * velocity[i] - learningRate * grad_with_decay;
            
            // Update parameters
            if (nesterov) {
                // Nesterov update
                param[i] += momentum * velocity[i] - learningRate * grad_with_decay;
            } else {
                // Standard momentum update
                param[i] += velocity[i];
            }
        }
    }
    
    // Update a scalar parameter
    void updateScalar(double &param,
                      double &velocity,
                      double &unused,
                      double grad) override {
        // Apply weight decay (usually not for bias)
        double grad_with_decay = grad;
        
        // Update velocity
        velocity = momentum * velocity - learningRate * grad_with_decay;
        
        // Update parameter
        if (nesterov) {
            // Nesterov update
            param += momentum * velocity - learningRate * grad_with_decay;
        } else {
            // Standard momentum update
            param += velocity;
        }
    }
    
    // Update a matrix parameter (using Eigen::MatrixXd)
    void updateMatrix(Eigen::MatrixXd& param,
                      Eigen::MatrixXd& velocity,
                      Eigen::MatrixXd& unused,
                      const Eigen::MatrixXd& grad) override {
        // Apply weight decay
        Eigen::MatrixXd grad_with_decay = grad + weightDecay * param;
        
        // Update velocity
        velocity = momentum * velocity - learningRate * grad_with_decay;
        
        // Update parameters
        if (nesterov) {
            // Nesterov update
            param += momentum * velocity - learningRate * grad_with_decay;
        } else {
            // Standard momentum update
            param += velocity;
        }
    }
    
    // Update a vector parameter (using Eigen::VectorXd)
    void updateVector(Eigen::VectorXd& param,
                      Eigen::VectorXd& velocity,
                      Eigen::VectorXd& unused,
                      const Eigen::VectorXd& grad) override {
        // Update velocity
        velocity = momentum * velocity - learningRate * grad;
        
        // Update parameter
        if (nesterov) {
            // Nesterov update
            param += momentum * velocity - learningRate * grad;
        } else {
            // Standard momentum update
            param += velocity;
        }
    }
};