#pragma once

#include "Layer.h"
#include <random>
#include <iostream>
#include "../common/Device.h"
#include "../cpu/CPUDevice.h"
#include "../metal/MetalDevice.h"


class DenseLayer : public Layer {
private:
    int inputDim;
    int outputDim;
    Device* device;

    Eigen::MatrixXd weights;
    Eigen::VectorXd biases;

    Eigen::MatrixXd m_w, v_w;
    Eigen::VectorXd m_b, v_b;

    double beta1 = 0.9;
    double beta2 = 0.999;
    double epsilon = 1e-8;
    int t = 0;

    double clipThreshold = 5.0;

    Eigen::MatrixXd lastInput;
    Eigen::MatrixXd lastOutput;

    Activation activation;
    ActivationType actType;

    double weightDecay = 0.0001;

public:
    DenseLayer(int outputDim, ActivationType actType = RELU, double weightDecay = 0.0001) 
        : outputDim(outputDim), actType(actType), activation(getActivation(actType)), weightDecay(weightDecay) {}

    void initialize(int inputDim, int outputDim, Device* device) override {
        this->inputDim = inputDim;
        this->outputDim = outputDim;
        this->device = device;
        
        double scale;
        if (actType == RELU || actType == LEAKY_RELU || actType == ELU) {
            scale = std::sqrt(2.0 / inputDim);
        
        } else {
            scale = std::sqrt(6.0 / (inputDim + outputDim));
        }

        std::random_device rd;
        std::mt19937 gen(rd());

        std::normal_distribution<double> dist(0.0, 1.0);

        weights = Eigen::MatrixXd::Zero(outputDim, inputDim);
        biases = Eigen::VectorXd::Zero(outputDim);

        for (int i = 0; i < outputDim; ++i) {
            for (int j = 0; j < inputDim; ++j) {
                weights(i, j) = dist(gen) * scale;
            }
        }

        m_w = Eigen::MatrixXd::Zero(outputDim, inputDim);
        v_w = Eigen::MatrixXd::Zero(outputDim, inputDim);
        m_b = Eigen::VectorXd::Zero(outputDim);
        v_b = Eigen::VectorXd::Zero(outputDim);
    }

    Eigen::MatrixXd forward(const Eigen::MatrixXd& input, bool isTraining) override {
        lastInput = input;

        Eigen::MatrixXd Z;
        if (dynamic_cast<MetalDevice*>(device)) {
            MetalDevice* gpu = dynamic_cast<MetalDevice*>(device);
            Z = gpu->matmulGPU(input, weights.transpose(), biases);
        } else {
            Z = input * weights.transpose();
            Z.rowwise() += biases.transpose();
        }

        if (actType != SOFTMAX) { 
            lastOutput = Z.unaryExpr([this](double z) { return activation.func(z); });
            return lastOutput;
        }
        else {

            lastOutput = Eigen::MatrixXd(Z.rows(), Z.cols());
            for (int i = 0; i < Z.rows(); i++) {
                double maxVal = Z.row(i).maxCoeff();
                Eigen::VectorXd exps = (Z.row(i).array() - maxVal).exp();
                double sumExps = exps.sum();
                lastOutput.row(i) = exps.transpose() / sumExps;
            }
        
            return lastOutput;
        }
    }

    Eigen::MatrixXd backward(const Eigen::MatrixXd& gradOutput, double learningRate) override {
        t++;
        int batchSize = lastInput.rows();

        Eigen::MatrixXd delta;
        if (actType != SOFTMAX) {
            Eigen::MatrixXd actDeriv = lastOutput.unaryExpr([this](double y) { return activation.derivative(y); });
            delta = gradOutput.cwiseProduct(actDeriv);
        } else {
            delta = gradOutput;
        }

        Eigen::MatrixXd dW = delta.transpose() * lastInput / batchSize;
        dW += weightDecay * weights;

        Eigen::VectorXd dB = delta.colwise().sum() / batchSize;

        // Gradient clipping
        double normDW = dW.norm();
        if (normDW > clipThreshold) {
            dW = (dW / normDW) * clipThreshold;
        }
        double normDB = dB.norm();
        if (normDB > clipThreshold) {
            dB = (dB / normDB) * clipThreshold;
        }
        // Update weights and biases using Adam

        m_w = beta1 * m_w + (1 - beta1) * dW;
        v_w = beta2 * v_w + (1 - beta2) * dW.cwiseProduct(dW);
        m_b = beta1 * m_b + (1 - beta1) * dB;
        v_b = beta2 * v_b + (1 - beta2) * dB.cwiseProduct(dB);

        double bias_correction1 = 1.0 / (1.0 - std::pow(beta1, t));
        double bias_correction2 = 1.0 / (1.0 - std::pow(beta2, t));

        Eigen::MatrixXd m_w_hat = m_w / bias_correction1;
        Eigen::MatrixXd v_w_hat = v_w / bias_correction2;
        Eigen::MatrixXd w_update = learningRate * m_w_hat.cwiseQuotient((v_w_hat.array().sqrt() + epsilon).matrix());
        weights -= w_update;

        Eigen::VectorXd m_b_hat = m_b / bias_correction1;
        Eigen::VectorXd v_b_hat = v_b / bias_correction2;
        Eigen::VectorXd b_update = learningRate * m_b_hat.cwiseQuotient((v_b_hat.array().sqrt() + epsilon).matrix());
        biases -= b_update;

        return delta * weights;

    }

    int getOutputSize() const override {
        return outputDim;
    }

    std::string getName() const override {
        return "DenseLayer";
    }

    void saveParameters(std::ostream& os) const override {
        for (int i = 0; i < outputDim; i++){
            for (int j = 0; j < inputDim; j++){
                os << weights(i, j) << " ";
            }
            os << "\n";
        }
        for (int i = 0; i < outputDim; i++){
            os << biases(i) << " ";
        }
        os << "\n";
    }

    void loadParameters(std::istream& is) override {
        for (int i = 0; i < outputDim; i++){
            for (int j = 0; j < inputDim; j++){
                is >> weights(i, j);
            }
        }
        for (int i = 0; i < outputDim; i++){
            is >> biases(i);
        }
    }

};