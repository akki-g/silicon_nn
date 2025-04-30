#include "common/Device.h"
#include "cpu/CPUDevice.h"
#include "metal/MetalDevice.h"
#include "parallel/parallel.h"
#include "optimizers/Adam.h"
#include "layers/Layer.h"
#include "layers/DenseLayer.h"
#include "layers/DropoutLayer.h"
#include "layers/BatchNormLayer.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <functional>
#include <random>
#include <algorithm>
#include <memory>
#include <chrono>
#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXi; // Add this line to properly include VectorXi

using namespace std;

// ---------------------------
// NeuralNetwork: Main class implementing the neural network
// ---------------------------
class NeuralNetwork {
private:
    vector<shared_ptr<Layer>> layers;
    Device* device;
    unique_ptr<Optimizer> optimizer;
    double learningRate;
    bool isTraining;
    int batchSize;
    ThreadPool pool;
    
    // Pre-allocated memory for training
    vector<MatrixXd> activations;    // Activations for each layer (including input)
    vector<MatrixXd> gradients;      // Gradients for each layer
    
    // Statistics for normalization
    VectorXd meanInput;
    VectorXd stdInput;
    bool isNormalized;
    
    // Metrics tracking
    vector<double> trainingLoss;
    vector<double> validationLoss;

public:
    NeuralNetwork(double lr = 0.001, char devType = 'c', double weightDecay = 0.0001, int batchSize = 32)
        : learningRate(lr), isTraining(true), batchSize(batchSize), 
          pool(std::thread::hardware_concurrency()), isNormalized(false)
    {
        if (devType == 'c')
            device = new CPUDevice();
        else {
            if (MetalDevice::metalIsAvailable()) {
                cout << "Using device: Metal" << endl;
                device = new MetalDevice();
            } else {
                cout << "Metal not available, defaulting to CPU." << endl;
                device = new CPUDevice();
            }
        }
        optimizer = make_unique<AdamW>(lr, weightDecay);
    }

    ~NeuralNetwork() {
        if (device) {
            device->cleanup();
            delete device;
        }
    }
    
    // Add a layer to the network
    void addLayer(shared_ptr<Layer> layer) {
        layers.push_back(layer);
    }
    
    // Add a Dense layer
    void addDenseLayer(int outputSize, ActivationType activation, bool useBatchNorm = false, double dropoutRate = 0.0) {
        int inputSize = layers.empty() ? 0 : layers.back()->getOutputSize();
        
        // Add the main dense layer
        auto dense = make_shared<DenseLayer>(outputSize, activation);
        if (!layers.empty()) {
            dense->initialize(inputSize, outputSize, device);
        }
        layers.push_back(dense);
        
        // Optionally add batch normalization
        if (useBatchNorm) {
            auto batchNorm = make_shared<BatchNormLayer>();
            batchNorm->initialize(outputSize, outputSize, device);
            layers.push_back(batchNorm);
        }
        
        // Optionally add dropout
        if (dropoutRate > 0.0) {
            auto dropout = make_shared<DropoutLayer>(dropoutRate);
            dropout->initialize(outputSize, outputSize, device);
            layers.push_back(dropout);
        }
    }
    
    // Initialize the first layer with input size
    void initializeFirstLayer(int inputSize) {
        if (!layers.empty() && dynamic_cast<DenseLayer*>(layers[0].get())) {
            auto denseLayer = dynamic_cast<DenseLayer*>(layers[0].get());
            denseLayer->initialize(inputSize, denseLayer->getOutputSize(), device);
        }
    }
    
    // Precompute statistics for input normalization
    void computeNormalizationStats(const MatrixXd& X) {
        // Calculate mean for each feature (column)
        meanInput = X.colwise().mean();
        
        // Calculate standard deviation for each feature
        stdInput = VectorXd(X.cols());
        for (int j = 0; j < X.cols(); j++) {
            // Calculate variance: mean of squared differences from mean
            double variance = 0.0;
            for (int i = 0; i < X.rows(); i++) {
                double diff = X(i, j) - meanInput(j);
                variance += diff * diff;
            }
            variance /= X.rows();
            
            // Standard deviation is square root of variance
            stdInput(j) = std::sqrt(variance);
            
            // Avoid division by zero
            if (stdInput(j) < 1e-8) {
                stdInput(j) = 1.0;
            }
        }
        
        isNormalized = true;
    }
    
    // Normalize input data
    MatrixXd normalizeInput(const MatrixXd& X) {
        if (!isNormalized) return X;
        
        MatrixXd normalized = MatrixXd::Zero(X.rows(), X.cols());
        
        // Normalize each feature
        for (int i = 0; i < X.rows(); i++) {
            for (int j = 0; j < X.cols(); j++) {
                normalized(i, j) = (X(i, j) - meanInput(j)) / stdInput(j);
            }
        }
        
        return normalized;
    }
    
    // Initialize memory for training
    void initializeMemory() {
        int numLayers = layers.size();
        activations.resize(numLayers + 1);  // +1 for input layer
        gradients.resize(numLayers);
    }

    // Forward pass through the network
    MatrixXd forward(const MatrixXd& X) {
        MatrixXd current = normalizeInput(X);
        
        // Store input as first activation
        if (isTraining && !activations.empty()) {
            activations[0] = current;
        }
        
        // Pass through each layer
        for (size_t i = 0; i < layers.size(); i++) {
            current = layers[i]->forward(current, isTraining);
            
            // Store activation for backpropagation
            if (isTraining && i + 1 < activations.size()) {
                activations[i + 1] = current;
            }
        }
        
        return current;
    }
    
    // Compute loss
    pair<double, MatrixXd> computeLoss(const MatrixXd& predictions, const MatrixXd& targets) {
        // For binary classification with sigmoid output
        if (predictions.cols() == 1) {
            // Binary cross-entropy loss
            MatrixXd loss = -(targets.array() * predictions.array().log() + 
                            (1 - targets.array()) * (1 - predictions.array()).log());
            double meanLoss = loss.mean();
            
            // Gradient: (predictions - targets)
            MatrixXd gradient = predictions - targets;
            
            return {meanLoss, gradient};
        }
        // For multi-class classification with softmax output
        else {
            // Cross-entropy loss
            MatrixXd loss(targets.rows(), 1);
            for (int i = 0; i < targets.rows(); i++) {
                int targetClass = 0;
                for (int j = 0; j < targets.cols(); j++) {
                    if (targets(i, j) > 0.5) {
                        targetClass = j;
                        break;
                    }
                }
                loss(i, 0) = -log(max(predictions(i, targetClass), 1e-7));
            }
            double meanLoss = loss.mean();
            
            // Gradient: (predictions - targets)
            MatrixXd gradient = predictions - targets;
            
            return {meanLoss, gradient};
        }
    }
    
    // Backward pass and update parameters
    void backward(const MatrixXd& lossGradient) {
        MatrixXd currentGradient = lossGradient;
        
        // Backpropagate through layers in reverse order
        for (int i = layers.size() - 1; i >= 0; i--) {
            currentGradient = layers[i]->backward(currentGradient, learningRate);
            
            if (isTraining && i < gradients.size()) {
                gradients[i] = currentGradient;
            }
        }
    }
    
    // Train on a single mini-batch
    double trainBatch(const MatrixXd& X, const MatrixXd& Y) {
        setTrainingMode(true);
        
        // Forward pass
        MatrixXd predictions = forward(X);
        
        // Compute loss and gradient
        auto [loss, gradient] = computeLoss(predictions, Y);
        
        // Backward pass
        backward(gradient);
        
        return loss;
    }
    
    // Fit the model to training data
    void fit(const MatrixXd& X, const MatrixXd& Y, int epochs, 
             const MatrixXd& X_val = MatrixXd(), const MatrixXd& Y_val = MatrixXd(), 
             int validationFrequency = 1) {
        
        // Compute normalization statistics
        computeNormalizationStats(X);
        
        // Initialize memory for training
        initializeMemory();
        
        // Initialize first layer if needed
        if (!layers.empty() && dynamic_cast<DenseLayer*>(layers[0].get())) {
            initializeFirstLayer(X.cols());
        }
        
        int numSamples = X.rows();
        vector<int> indices(numSamples);
        for (int i = 0; i < numSamples; i++) {
            indices[i] = i;
        }
        
        // Initialize metrics tracking
        trainingLoss.clear();
        validationLoss.clear();
        
        // Training loop
        for (int epoch = 0; epoch < epochs; epoch++) {
            auto startTime = chrono::high_resolution_clock::now();
            
            // Shuffle indices for stochastic training
            shuffle(indices.begin(), indices.end(), mt19937(random_device()()));
            
            double epochLoss = 0.0;
            int numBatches = 0;
            
            // Mini-batch training
            for (int batchStart = 0; batchStart < numSamples; batchStart += batchSize) {
                int batchEnd = min(batchStart + batchSize, numSamples);
                int currentBatchSize = batchEnd - batchStart;
                
                // Create batch matrices
                MatrixXd batchX(currentBatchSize, X.cols());
                MatrixXd batchY(currentBatchSize, Y.cols());
                
                // Fill batch matrices with shuffled data
                for (int i = 0; i < currentBatchSize; i++) {
                    int idx = indices[batchStart + i];
                    batchX.row(i) = X.row(idx);
                    batchY.row(i) = Y.row(idx);
                }
                
                // Train on this batch
                double batchLoss = trainBatch(batchX, batchY);
                epochLoss += batchLoss;
                numBatches++;
            }
            
            // Calculate average loss for this epoch
            epochLoss /= numBatches;
            trainingLoss.push_back(epochLoss);
            
            // Validation if requested
            double valLoss = 0.0;
            if (X_val.rows() != 0 && epoch % validationFrequency == 0) {
                setTrainingMode(false);
                MatrixXd predictions = forward(X_val);
                auto [loss, _] = computeLoss(predictions, Y_val);
                valLoss = loss;
                validationLoss.push_back(valLoss);
            }
            
            auto endTime = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::milliseconds>(endTime - startTime).count();
            
            // Print progress
            cout << "Epoch " << epoch + 1 << "/" << epochs 
                 << " - loss: " << epochLoss;
            
            if (X_val.rows() != 0 && epoch % validationFrequency == 0) {
                cout << " - val_loss: " << valLoss;
            }
            
            cout << " - time: " << duration << "ms" << endl;
        }
    }
    
    // Predict output for new data
    MatrixXd predict(const MatrixXd& X) {
        setTrainingMode(false);
        return forward(X);
    }
    
    // Make a binary prediction (for binary classification)
    VectorXd predictBinary(const MatrixXd& X, double threshold = 0.5) {
        MatrixXd probs = predict(X);
        VectorXd predictions(X.rows());
        
        for (int i = 0; i < X.rows(); i++) {
            predictions(i) = probs(i, 0) > threshold ? 1.0 : 0.0;
        }
        
        return predictions;
    }
    
    // Make a class prediction (for multi-class classification)
    VectorXi predictClass(const MatrixXd& X) {
        MatrixXd probs = predict(X);
        VectorXi predictions(X.rows());
        
        for (int i = 0; i < X.rows(); i++) {
            int maxIdx = 0;
            double maxVal = probs(i, 0);
            
            for (int j = 1; j < probs.cols(); j++) {
                if (probs(i, j) > maxVal) {
                    maxVal = probs(i, j);
                    maxIdx = j;
                }
            }
            
            predictions(i) = maxIdx;
        }
        
        return predictions;
    }
    
    // Calculate accuracy for binary classification
    double accuracy(const MatrixXd& X, const MatrixXd& Y, double threshold = 0.5) {
        VectorXd pred = predictBinary(X, threshold);
        VectorXd targets = Y.col(0);
        
        int correct = 0;
        for (int i = 0; i < X.rows(); i++) {
            if (pred(i) == targets(i)) {
                correct++;
            }
        }
        
        return static_cast<double>(correct) / X.rows();
    }
    
    // Calculate accuracy for multi-class classification
    double accuracyMultiClass(const MatrixXd& X, const MatrixXd& Y) {
        VectorXi pred = predictClass(X);
        VectorXi targets(Y.rows());
        
        for (int i = 0; i < Y.rows(); i++) {
            for (int j = 0; j < Y.cols(); j++) {
                if (Y(i, j) > 0.5) {
                    targets(i) = j;
                    break;
                }
            }
        }
        
        int correct = 0;
        for (int i = 0; i < X.rows(); i++) {
            if (pred(i) == targets(i)) {
                correct++;
            }
        }
        
        return static_cast<double>(correct) / X.rows();
    }
    
    // Set training mode (affects dropout, batch norm, etc.)
    void setTrainingMode(bool training) {
        isTraining = training;
    }

    // Get training loss history
    const vector<double>& getTrainingLoss() const {
        return trainingLoss;
    }
    
    // Get validation loss history
    const vector<double>& getValidationLoss() const {
        return validationLoss;
    }
    
    // Clean up device resources
    void cleanDevice() {
        if (device) {
            device->cleanup();
        }
    }
};

//////////////////////////////
// Extern "C" Interface (for Python bindings)
//////////////////////////////
extern "C" {

NeuralNetwork* createNN(double learningRate, char deviceType) {
    srand((unsigned)time(0));
    return new NeuralNetwork(learningRate, deviceType);
}

void destroyNN(NeuralNetwork* nn) {
    delete nn;
}

void addLayerNN(NeuralNetwork* nn, int numNeurons, int inputSize, int activationType) {
    nn->initializeFirstLayer(inputSize);
    nn->addDenseLayer(numNeurons, static_cast<ActivationType>(activationType));
}

void addHiddenLayerNN(NeuralNetwork* nn, int numNeurons, int activationType) {
    nn->addDenseLayer(numNeurons, static_cast<ActivationType>(activationType));
}

void addOutputLayerNN(NeuralNetwork* nn, int numNeurons, int activationType) {
    nn->addDenseLayer(numNeurons, static_cast<ActivationType>(activationType), false, 0.0);
}

// Add a layer with batch normalization
void addBatchNormLayerNN(NeuralNetwork* nn, int numNeurons, int activationType) {
    nn->addDenseLayer(numNeurons, static_cast<ActivationType>(activationType), true, 0.0);
}

// Add a layer with dropout
void addDropoutLayerNN(NeuralNetwork* nn, int numNeurons, int activationType, double dropRate) {
    nn->addDenseLayer(numNeurons, static_cast<ActivationType>(activationType), false, dropRate);
}

void fitNN(NeuralNetwork* nn, double* inputs, int numSamples, int inputSize,
           double* targets, int targetSize, int epochs, int batchSize) {
    // Convert C arrays to Eigen matrices
    MatrixXd X(numSamples, inputSize);
    MatrixXd Y(numSamples, targetSize);
    
    for (int i = 0; i < numSamples; i++) {
        for (int j = 0; j < inputSize; j++) {
            X(i, j) = inputs[i * inputSize + j];
        }
        for (int j = 0; j < targetSize; j++) {
            Y(i, j) = targets[i * targetSize + j];
        }
    }
    
    nn->fit(X, Y, epochs);
}

void predictNN(NeuralNetwork* nn, double* input, int inputSize, double* output, int outputSize) {
    // Convert single input to Eigen row vector
    MatrixXd X(1, inputSize);
    for (int j = 0; j < inputSize; j++) {
        X(0, j) = input[j];
    }
    
    // Get prediction
    MatrixXd pred = nn->predict(X);
    
    // Copy prediction to output array
    for (int j = 0; j < min(outputSize, (int)pred.cols()); j++) {
        output[j] = pred(0, j);
    }
}

void setTrainingModeNN(NeuralNetwork* nn, int mode) {
    nn->setTrainingMode(mode != 0);
}

void cleanDeviceNN(NeuralNetwork* nn) {
    if (nn) {
        nn->cleanDevice();
    }
}

} // extern "C"