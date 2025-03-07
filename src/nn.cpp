#include "common/Device.h"
#include "cpu/CPUDevice.h"
#include "metal/MetalDevice.h"
#include "parallel/parallel.h"
#include "optimizers/Adam.h"  // Contains Optimizer and AdamW definitions

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <functional>
#include <random>
#include <algorithm>

using namespace std;

// Activation functions
enum ActivationType {
    SIGMOID = 0, 
    RELU = 1,
    TANH = 2
};

struct Activation {
    function<double(double)> func;
    function<double(double)> derivative;
};

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}
double sigmoidDerivative(double activatedValue) {
    return activatedValue * (1.0 - activatedValue);
}

double relu(double x) {
    return x > 0 ? x : 0.0;
}
double reluDerivative(double activatedValue) {
    return activatedValue > 0 ? 1.0 : 0.0;
}

double tanhActivation(double x) {
    return tanh(x);
}
double tanhDerivative(double activatedValue) {
    return 1.0 - activatedValue * activatedValue;
}

Activation getActivation(ActivationType type) {
    switch (type) {
        case SIGMOID: return {sigmoid, sigmoidDerivative};
        case RELU:    return {relu, reluDerivative};
        case TANH:    return {tanhActivation, tanhDerivative};
        default:
            cerr << "Invalid activation function type" << endl;
            exit(1);
    }
}

// Dropout function
vector<double> applyDropout(const vector<double>& activations, double dropoutProb) {
    vector<double> output(activations.size());
    for (size_t i = 0; i < activations.size(); i++) {
        double randomVal = ((double) rand() / RAND_MAX);
        if (randomVal < dropoutProb)
            output[i] = 0.0;
        else
            output[i] = activations[i] / (1.0 - dropoutProb);
    }
    return output;
}

namespace parallel {
    inline ThreadPool globalThreadPool(std::thread::hardware_concurrency());
}

// Neuron class with Adam accumulators.
class Neuron {
public: 
    vector<double> weights;
    double bias;
    double output;
    double error;

    vector<double> m_weights, v_weights;
    double m_bias, v_bias;

    Neuron(int numInputs) {
        double limit = sqrt(6.0 / numInputs);
        uniform_real_distribution<double> dist(-limit, limit);
        default_random_engine engine(random_device{}());
        for (int i = 0; i < numInputs; i++){
            double w = dist(engine);
            weights.push_back(w);
            m_weights.push_back(0.0);
            v_weights.push_back(0.0);
        }
        bias = dist(engine);
        m_bias = 0.0;
        v_bias = 0.0;
        output = 0.0;
    }

    // Activation function using the device's dot function.
    double activate(vector<double>& inputs, const Activation& act, Device* device) {
        double sum = device->dot(inputs, weights) + bias;
        output = act.func(sum);
        return output;
    }
};

// DenseLayer class.
class DenseLayer {
public:
    vector<Neuron> neurons;
    Activation activationFunction;

    DenseLayer(int numNeurons, int numInputs, const Activation& act = {sigmoid, sigmoidDerivative})
        : activationFunction(act) {
        for (int i = 0; i < numNeurons; i++) {
            neurons.push_back(Neuron(numInputs));
        }
    }

    // Forward pass using parallel_for.
    vector<double> forward(const vector<double>& inputs, Device* device) {
        size_t n = neurons.size();
        vector<double> outputs(n);
        size_t blockSize = (n < 64) ? n : 32;  // Tunable parameter.
        size_t totalBlocks = (n + blockSize - 1) / blockSize;
        parallel_for<size_t>(0, totalBlocks, [&](size_t block) {
            size_t start = block * blockSize;
            size_t end = min(start + blockSize, n);
            for (size_t i = start; i < end; i++) {
                outputs[i] = neurons[i].activate(const_cast<vector<double>&>(inputs), activationFunction, device);
            }
        }, parallel::globalThreadPool);
        return outputs;
    }
};

// BatchNormLayer class.
class BatchNormLayer {
public: 
    double gamma, beta;
    double runningMean, runningVar;
    double momentum;
    double epsilon;
    
    BatchNormLayer(double momentum = 0.9, double epsilon = 1e-5)
        : gamma(1.0), beta(0.0), runningMean(0.0), runningVar(1.0),
          momentum(momentum), epsilon(epsilon) {}

    vector<double> forward(const vector<double>& x) {
        double mean = computeMean(x);
        double var = computeVariance(x, mean);
        runningMean = momentum * runningMean + (1 - momentum) * mean;
        runningVar = momentum * runningVar + (1 - momentum) * var;
        vector<double> normalized(x.size());
        size_t n = x.size();
        size_t blockSize = (n < 64) ? n : 32;
        size_t totalBlocks = (n + blockSize - 1) / blockSize;
        parallel_for<size_t>(0, totalBlocks, [&](size_t block) {
            size_t start = block * blockSize;
            size_t end = min(start + blockSize, n);
            for (size_t i = start; i < end; i++) {
                normalized[i] = gamma * ((x[i] - mean) / sqrt(var + epsilon)) + beta;
            }
        }, parallel::globalThreadPool);
        return normalized;
    }
private: 
    double computeMean(const vector<double>& x) {
        double sum = 0.0;
        for (double val : x)
            sum += val;
        return sum / x.size();
    }
    double computeVariance(const vector<double>& x, double mean) {
        double sum = 0.0;
        for (double val : x)
            sum += (val - mean) * (val - mean);
        return sum / x.size();
    }
};

// ResidualBlock class.
class ResidualBlock {
public:
    DenseLayer layer1, layer2;

    ResidualBlock(int numNeurons, int numInputs, const Activation& act)
        : layer1(numNeurons, numInputs, act), layer2(numNeurons, numNeurons, act) {}

    vector<double> forward(const vector<double>& input, Device* device) {
        vector<double> out1 = layer1.forward(input, device);
        vector<double> out2 = layer2.forward(out1, device);
        if (input.size() != out2.size()) {
            cerr << "Residual block input and output size mismatch" << endl;
            exit(1);
        }
        vector<double> output(input.size());

        size_t n = input.size();
        size_t blockSize = (n < 64) ? n : 32;
        size_t totalBlocks = (n + blockSize - 1) / blockSize;
        parallel_for<size_t>(0, totalBlocks, [&](size_t block) {
            size_t start = block * blockSize;
            size_t end = min(start + blockSize, n);
            for (size_t i = start; i < end; i++) {
                output[i] = input[i] + out2[i];
            }
        }, parallel::globalThreadPool);
        return output;
    }
};

// NeuralNetwork class with mini-batch training using AdamW and regularization.
class NeuralNetwork {
public:
    vector<DenseLayer> layers;
    double learningRate;
    vector<BatchNormLayer> batchNormLayers;
    bool useBatchNorm;
    bool useDropout;
    double dropoutProb;
    ResidualBlock* residualBlock;
    bool useResidualBlock;
    Device* device;
    Optimizer* optimizer;  // Modular optimizer pointer

    NeuralNetwork(double lr = 0.001, char c = 'c', bool batch = true, bool dropout = true,
                  double dropProb = 0.05, bool residual = true, double weightDecay = 0.0001)
        : learningRate(lr), useBatchNorm(batch), useDropout(dropout), dropoutProb(dropProb),
          useResidualBlock(residual), residualBlock(nullptr) {
        if (c == 'c') {
            device = new CPUDevice();
        } else {
            if (MetalDevice::metalIsAvailable()) {
                cout << "Using device: Metal" << endl;
                device = new MetalDevice();
            } else {
                cout << "Metal is not available, defaulting to CPU" << endl;
                device = new CPUDevice();
            }
        }
        // Create an AdamW optimizer with decoupled weight decay.
        optimizer = new AdamW(lr, weightDecay);
    }

    ~NeuralNetwork() {
        delete device;
        if (residualBlock) delete residualBlock;
        delete optimizer;
    }

    void initBatchNormLayers(int numLayers) {
        for (int i = 0; i < numLayers; i++) {
            batchNormLayers.push_back(BatchNormLayer());
        }
    }

    void initResidualBlock(int numNeurons, int numInputs, const Activation& act) {
        if (residualBlock) { delete residualBlock; }
        residualBlock = new ResidualBlock(numNeurons, numInputs, act);
    }

    void addLayer(int numNeurons, int inputSize, const Activation& act) {
        if (!layers.empty()){
            cerr << "First layer already added." << endl;
            exit(1);
        }
        layers.push_back(DenseLayer(numNeurons, inputSize, act));
    }

    void addLayer(int numNeurons, Activation act) {
        if (layers.empty()){
            cerr << "Input size must be provided for the first layer" << endl;
            exit(1);
        }
        int prevLayerSize = layers.back().neurons.size();
        layers.push_back(DenseLayer(numNeurons, prevLayerSize, act));
    }

    // Compute outputs for every layer.
    vector<vector<double>> forwardAllLayers(const vector<double>& sample) {
        vector<vector<double>> layerOutputs;
        layerOutputs.push_back(sample);
        vector<double> current = sample;
        for (size_t i = 0; i < layers.size(); i++) {
            current = layers[i].forward(current, device);
            if (useBatchNorm && i < batchNormLayers.size())
                current = batchNormLayers[i].forward(current);
            if (useDropout)
                current = applyDropout(current, dropoutProb);
            if (useResidualBlock && (i == 0) && residualBlock != nullptr)
                current = residualBlock->forward(current, device);
            layerOutputs.push_back(current);
        }
        return layerOutputs;
    }

    // Mini-batch training function.
    // --- Inside NeuralNetwork class ---

    // Mini-batch training function with corrected gradient accumulation.
    void trainBatch(const vector<vector<double>>& batchInputs,
                    const vector<vector<double>>& batchTargets) {
        size_t batchSize = batchInputs.size();
        size_t numLayers = layers.size();

        // Forward pass for each sample.
        vector<vector<vector<double>>> allLayerOutputs(batchSize);
        for (size_t s = 0; s < batchSize; s++) {
            allLayerOutputs[s] = forwardAllLayers(batchInputs[s]);
        }

        // Initialize gradient accumulators:
        // gradAcc[l][i] will be a vector (of size equal to input dimension for layer l)
        // that accumulates the gradient for neuron i in layer l.
        vector<vector<vector<double>>> gradAcc(numLayers);
        for (size_t l = 0; l < numLayers; l++) {
            size_t numNeurons = layers[l].neurons.size();
            // Determine input size for layer l: for l==0, it is batchInputs[0].size(); else, the number of neurons in the previous layer.
            size_t inputSize = (l == 0) ? batchInputs[0].size() : layers[l-1].neurons.size();
            gradAcc[l].resize(numNeurons, vector<double>(inputSize, 0.0));
        }
        
        // Initialize bias gradient accumulators for each neuron.
        vector<vector<double>> gradBias(numLayers);
        for (size_t l = 0; l < numLayers; l++) {
            gradBias[l].resize(layers[l].neurons.size(), 0.0);
        }

        // Backward pass: for each sample, compute per-neuron error and accumulate weight gradients.
        // We'll store sampleErrors[s][l][i] = error for neuron i in layer l for sample s.
        vector<vector<vector<double>>> sampleErrors(batchSize);
        for (size_t s = 0; s < batchSize; s++) {
            sampleErrors[s].resize(numLayers);
            // Output layer error.
            int outLayer = numLayers - 1;
            size_t outCount = layers[outLayer].neurons.size();
            sampleErrors[s][outLayer] = vector<double>(outCount, 0.0);
            for (size_t i = 0; i < outCount; i++) {
                double outVal = allLayerOutputs[s][outLayer][i];
                sampleErrors[s][outLayer][i] = outVal - batchTargets[s][i];
            }
            // Backpropagate errors from last layer to first.
            for (int l = numLayers - 2; l >= 0; l--) {
                size_t neuronCount = layers[l].neurons.size();
                sampleErrors[s][l] = vector<double>(neuronCount, 0.0);
                for (size_t i = 0; i < neuronCount; i++) {
                    double errorSum = 0.0;
                    for (size_t j = 0; j < layers[l+1].neurons.size(); j++) {
                        errorSum += layers[l+1].neurons[j].weights[i] * sampleErrors[s][l+1][j];
                    }
                    double outVal = allLayerOutputs[s][l+1][i];
                    sampleErrors[s][l][i] = errorSum * layers[l].activationFunction.derivative(outVal);
                }
            }
            
            // Accumulate gradients for each layer.
            for (size_t l = 0; l < numLayers; l++) {
                // Determine the input vector for layer l: if l==0, use batchInputs[s]; else, use output of previous layer.
                const vector<double>& inputVec = (l == 0) ? batchInputs[s] : allLayerOutputs[s][l];
                size_t inputSize = inputVec.size();
                size_t numNeurons = layers[l].neurons.size();
                for (size_t i = 0; i < numNeurons; i++) {
                    // For each weight in neuron i, accumulate the product error * input.
                    for (size_t j = 0; j < inputSize; j++) {
                        gradAcc[l][i][j] += sampleErrors[s][l][i] * inputVec[j];
                    }
                    // Accumulate bias gradient.
                    gradBias[l][i] += sampleErrors[s][l][i];
                }
            }
        }

        // Average gradients over the batch and update parameters via the optimizer.
        for (size_t l = 0; l < numLayers; l++) {
            size_t numNeurons = layers[l].neurons.size();
            for (size_t i = 0; i < numNeurons; i++) {
                // Average weight gradients.
                vector<double> avgGrad = gradAcc[l][i];
                for (double &val : avgGrad)
                    val /= batchSize;
                double avgBiasGrad = gradBias[l][i] / batchSize;
                // Update weights and bias using the optimizer.
                optimizer->updateVector(layers[l].neurons[i].weights,
                                        layers[l].neurons[i].m_weights,
                                        layers[l].neurons[i].v_weights,
                                        avgGrad);
                optimizer->updateScalar(layers[l].neurons[i].bias,
                                        layers[l].neurons[i].m_bias,
                                        layers[l].neurons[i].v_bias,
                                        avgBiasGrad);
            }
        }
    }


    // Single-sample training wrapper.
    void train(const vector<double>& input, const vector<double>& target) {
        vector<vector<double>> batchInput = { input };
        vector<vector<double>> batchTarget = { target };
        trainBatch(batchInput, batchTarget);
    }

    // Forward pass for a single sample.
    vector<double> forward(vector<double>& input) {
        return forwardAllLayers(input).back();
    }
};

#ifdef __cplusplus
extern "C" {
#endif

NeuralNetwork* createNN(double learningRate, char c) {
    srand((unsigned) time(0));
    return new NeuralNetwork(learningRate, c);
}

void destroyNN(NeuralNetwork* nn) {
    delete nn;
}

void initBatchNormLayers(NeuralNetwork* nn, int numLayers) {
    nn->initBatchNormLayers(numLayers);
}

void initResidualBlock(NeuralNetwork* nn, int numNeurons, int numInputs, int activationType) {
    Activation act = getActivation(static_cast<ActivationType>(activationType));
    nn->initResidualBlock(numNeurons, numInputs, act);
}

void addLayerNN(NeuralNetwork* nn, int numNeurons, int inputSize, int activationType) {
    Activation act = getActivation(static_cast<ActivationType>(activationType));
    nn->addLayer(numNeurons, inputSize, act);
}

void addLayerNN_noInput(NeuralNetwork* nn, int numNeurons, int activationType) {
    Activation act = getActivation(static_cast<ActivationType>(activationType));
    nn->addLayer(numNeurons, act);
}

void trainSampleNN(NeuralNetwork* nn, double* input, int inputSize, double* target, int targetSize) {
    vector<double> inputVec(input, input + inputSize);
    vector<double> targetVec(target, target + targetSize);
    nn->train(inputVec, targetVec);
}


void fitNN(NeuralNetwork* nn, double* inputs, int numSample, int inputSize, double* targets, int targetSize, int epochs) {
    // Create an index vector [0, 1, 2, ..., numSample-1]
    std::vector<int> indices(numSample);
    for (int i = 0; i < numSample; i++) {
        indices[i] = i;
    }

    // Setup a random engine for shuffling
    std::random_device rd;
    std::mt19937 g(rd());

    for (int epoch = 0; epoch < epochs; epoch++) {
        // Shuffle the indices for each epoch
        std::shuffle(indices.begin(), indices.end(), g);

        for (int j = 0; j < numSample; j++) {
            // Use the shuffled index
            int idx = indices[j];
            double* inputStart = inputs + idx * inputSize;
            double* targetStart = targets + idx * targetSize;
            std::vector<double> inputVec(inputStart, inputStart + inputSize);
            std::vector<double> targetVec(targetStart, targetStart + targetSize);
            nn->train(inputVec, targetVec);
        }
    }
}

void predictNN(NeuralNetwork* nn, double* input, int inputSize, double* output, int outputSize) {
    vector<double> inputVec(input, input + inputSize);
    vector<double> result = nn->forward(inputVec);
    int len = result.size();
    if (len > outputSize)
        len = outputSize;
    for (int i = 0; i < len; i++) {
        output[i] = result[i];
    }
}


void evaluationMetrics(NeuralNetwork &nn, vector<vector<double>> &inputs, vector<vector<double>> &targets) {
    int TP = 0, TN = 0, FP = 0, FN = 0;
    for (size_t i = 0; i < inputs.size(); i++) {
        vector<double> output = nn.forward(inputs[i]);
        int predicted = output[0] > 0.5 ? 1 : 0;
        int actual = targets[i][0];
        if (predicted == 1 && actual == 1) TP++;
        else if (predicted == 0 && actual == 0) TN++;
        else if (predicted == 1 && actual == 0) FP++;
        else if (predicted == 0 && actual == 1) FN++;
    }
    double accuracy = (double)(TP + TN) / (TP + TN + FP + FN);
    double precision = (TP + FP > 0) ? (double)TP / (TP + FP) : 0.0;
    double recall = (TP + FN > 0) ? (double)TP / (TP + FN) : 0.0;
    double f1 = (precision + recall > 0) ? 2 * precision * recall / (precision + recall) : 0.0;
    cout << "\n--- Evaluation Metrics ---\n";
    cout << "Accuracy:  " << accuracy << "\n";
    cout << "Precision: " << precision << "\n";
    cout << "Recall:    " << recall << "\n";
    cout << "F1 Score:  " << f1 << "\n";
}

void cleanupDevice(NeuralNetwork* nn) {
    if (nn && nn->device) {
        nn->device->cleanup();
    }
}

#ifdef __cplusplus
}
#endif
