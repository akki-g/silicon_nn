// Optimizer.h
#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <vector>
#include <cmath>
#include <stdexcept>
#include <functional>

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
    virtual ~Optimizer() {}
};

// AdamW optimizer: decoupled weight decay.
class AdamW : public Optimizer {
public:
    double learningRate;
    double beta1;
    double beta2;
    double epsilon;
    double weightDecay; // decoupled weight decay

    AdamW(double lr, double wd, double b1 = 0.9, double b2 = 0.999, double eps = 1e-8)
      : learningRate(lr), weightDecay(wd), beta1(b1), beta2(b2), epsilon(eps) {}

    // Update a vector parameter (e.g. weights)
    virtual void updateVector(std::vector<double>& param,
                              std::vector<double>& m,
                              std::vector<double>& v,
                              const std::vector<double>& grad) override {
        for (size_t i = 0; i < param.size(); i++) {
            m[i] = beta1 * m[i] + (1 - beta1) * grad[i];
            v[i] = beta2 * v[i] + (1 - beta2) * grad[i] * grad[i];
            double update = learningRate * m[i] / (std::sqrt(v[i]) + epsilon);
            // Apply decoupled weight decay
            param[i] -= update + learningRate * weightDecay * param[i];
        }
    }

    // Update a scalar parameter (e.g. bias); typically weight decay is not applied.
    virtual void updateScalar(double &param,
                              double &m,
                              double &v,
                              double grad) override {
        m = beta1 * m + (1 - beta1) * grad;
        v = beta2 * v + (1 - beta2) * grad * grad;
        double update = learningRate * m / (std::sqrt(v) + epsilon);
        param -= update;
    }
};

#endif // OPTIMIZER_H
