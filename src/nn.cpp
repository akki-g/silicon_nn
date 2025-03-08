#include "common/Device.h"
#include "cpu/CPUDevice.h"
#include "metal/MetalDevice.h"
#include "parallel/parallel.h"
#include "optimizers/Adam.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <functional>
#include <random>
#include <algorithm>

using namespace std;

// ---------------------------
// Activation Functions & Types
// ---------------------------
enum ActivationType { SIGMOID = 0, RELU = 1, TANH = 2, SOFTMAX = 3 };

struct Activation {
    function<double(double)> func;
    function<double(double)> derivative;
};

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

// For softmax, we will handle the activation in the forward pass.
// Here we just use identity (the derivative is built into the cross-entropy loss).
Activation getActivation(ActivationType type) {
    switch (type) {
        case SIGMOID: return { sigmoid, sigmoidDerivative };
        case RELU:    return { relu, reluDerivative };
        case TANH:    return { tanhActivation, tanhDerivative };
        case SOFTMAX: return { [](double x){ return x; }, [](double y){ return 1.0; } };
        default:
            cerr << "Invalid activation type." << endl;
            exit(1);
    }
}

// ---------------------------
// Dropout (applied during forward pass)
// ---------------------------
void applyDropout(vector<double>& activations, double dropoutProb) {
    for (size_t i = 0; i < activations.size(); i++) {
        double r = (double)rand() / RAND_MAX;
        if (r < dropoutProb)
            activations[i] = 0.0;
        else
            activations[i] /= (1.0 - dropoutProb);
    }
}

// ---------------------------
// Neuron: stores weights, bias, and Adam accumulators
// ---------------------------
class Neuron {
public:
    vector<double> weights;
    double bias;
    vector<double> m_w, v_w;  // For Adam optimizer (weights)
    double m_b, v_b;          // For Adam optimizer (bias)
    double output;            // Latest output after activation

    Neuron(int numInputs) {
        // Xavier initialization (using fan_in)
        double limit = sqrt(6.0 / numInputs);
        static default_random_engine engine((unsigned)time(NULL));
        uniform_real_distribution<double> dist(-limit, limit);
        weights.resize(numInputs);
        m_w.resize(numInputs, 0.0);
        v_w.resize(numInputs, 0.0);
        for (int i = 0; i < numInputs; i++) {
            weights[i] = dist(engine);
        }
        bias = dist(engine);
        m_b = 0.0;
        v_b = 0.0;
        output = 0.0;
    }
};

// ---------------------------
// DenseLayer: a fully connected layer
// ---------------------------
class DenseLayer {
public:
    vector<Neuron> neurons;
    Activation activation;
    ActivationType actType;

    DenseLayer(int numNeurons, int numInputs, ActivationType actType_ = SIGMOID)
        : activation(getActivation(actType_)), actType(actType_)
    {
        for (int i = 0; i < numNeurons; i++) {
            neurons.push_back(Neuron(numInputs));
        }
    }

    // Forward pass: compute each neuron's output
    vector<double> forward(const vector<double>& inputs, Device* device) {
        size_t numNeurons = neurons.size();
        vector<double> outputs(numNeurons, 0.0);
        if (actType == SOFTMAX) {
            // Compute logits first
            vector<double> logits(numNeurons, 0.0);
            for (size_t i = 0; i < numNeurons; i++) {
                logits[i] = device->dot(inputs, neurons[i].weights) + neurons[i].bias;
            }
            // Softmax computation for numerical stability
            double maxLogit = *max_element(logits.begin(), logits.end());
            double sumExp = 0.0;
            for (size_t i = 0; i < numNeurons; i++) {
                sumExp += exp(logits[i] - maxLogit);
            }
            for (size_t i = 0; i < numNeurons; i++) {
                outputs[i] = exp(logits[i] - maxLogit) / sumExp;
                neurons[i].output = outputs[i];
            }
        } else {
            for (size_t i = 0; i < numNeurons; i++) {
                double z = device->dot(inputs, neurons[i].weights) + neurons[i].bias;
                double a = activation.func(z);
                neurons[i].output = a;
                outputs[i] = a;
            }
        }
        return outputs;
    }
};

// ---------------------------
// NeuralNetwork: main class implementing forward and backpropagation
// ---------------------------
class NeuralNetwork {
public:
    vector<DenseLayer> layers;
    Device* device;
    Optimizer* optimizer;
    double learningRate;
    bool useDropout;
    double dropoutProb;
    bool isTraining;

    NeuralNetwork(double lr = 0.001, char devType = 'c', bool dropout = true, double dropProb = 0.1, double weightDecay = 0.0001)
        : learningRate(lr), useDropout(dropout), dropoutProb(dropProb), isTraining(true)
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
        optimizer = new AdamW(lr, weightDecay);
    }

    ~NeuralNetwork() {
        delete device;
        delete optimizer;
    }

    // Add the first layer (requires input size)
    void addLayer(int numNeurons, int inputSize, ActivationType actType) {
        layers.push_back(DenseLayer(numNeurons, inputSize, actType));
    }

    // Add subsequent hidden or output layers
    void addLayer(int numNeurons, ActivationType actType) {
        if (layers.empty()) {
            cerr << "Error: First layer must specify input size." << endl;
            exit(1);
        }
        int prevSize = layers.back().neurons.size();
        layers.push_back(DenseLayer(numNeurons, prevSize, actType));
    }

    // Compute forward pass through all layers and store activations
    // activations[0] is the input; activations[i] (i>=1) is the output from layer i-1.
    vector< vector<double> > forwardAllLayers(const vector<double>& input) {
        vector< vector<double> > activations;
        activations.push_back(input); // input layer
        vector<double> current = input;
        for (size_t l = 0; l < layers.size(); l++) {
            current = layers[l].forward(current, device);
            // Apply dropout on hidden layers (skip output layer)
            if (isTraining && useDropout && l < layers.size() - 1) {
                applyDropout(current, dropoutProb);
            }
            activations.push_back(current);
        }
        return activations;
    }

    // Mini-batch training: perform forward and backward pass on a batch and update parameters.
    void trainBatch(const vector< vector<double> >& batchInputs,
                    const vector< vector<double> >& batchTargets)
    {
        size_t batchSize = batchInputs.size();
        int L = layers.size();

        // Initialize accumulators for gradients:
        // gradW[l][i][j] is the gradient for weight j of neuron i in layer l.
        vector< vector< vector<double> > > gradW(L);
        vector< vector<double> > gradB(L);
        for (int l = 0; l < L; l++) {
            int numNeurons = layers[l].neurons.size();
            int inputSize = (l == 0) ? batchInputs[0].size() : layers[l - 1].neurons.size();
            gradW[l].resize(numNeurons, vector<double>(inputSize, 0.0));
            gradB[l].resize(numNeurons, 0.0);
        }

        // Process each sample in the batch
        for (size_t s = 0; s < batchSize; s++) {
            // Forward pass: store activations from input to output
            vector< vector<double> > activations = forwardAllLayers(batchInputs[s]);
            // Backward pass: compute deltas for each layer
            vector< vector<double> > deltas(L);

            // --- Output layer delta ---
            int outLayer = L - 1;
            int outSize = activations.back().size(); // final output size
            deltas[outLayer].resize(outSize, 0.0);
            for (int i = 0; i < outSize; i++) {
                double a = activations.back()[i];
                double t = batchTargets[s][i];
                // For cross-entropy loss with sigmoid/softmax output, delta = (a - t)
                deltas[outLayer][i] = a - t;
            }

            // --- Hidden layers deltas (backpropagation) ---
            for (int l = L - 2; l >= 0; l--) {
                int layerSize = layers[l].neurons.size();
                deltas[l].resize(layerSize, 0.0);
                for (int i = 0; i < layerSize; i++) {
                    double errorSum = 0.0;
                    int nextLayerNeurons = layers[l + 1].neurons.size();
                    for (int j = 0; j < nextLayerNeurons; j++) {
                        errorSum += layers[l + 1].neurons[j].weights[i] * deltas[l + 1][j];
                    }
                    // Use the activation output from the current layer (activations[l+1])
                    double a = activations[l + 1][i];
                    deltas[l][i] = errorSum * layers[l].activation.derivative(a);
                }
            }

            // --- Accumulate gradients for each layer ---
            for (int l = 0; l < L; l++) {
                int numNeurons = layers[l].neurons.size();
                int inputSize = (l == 0) ? batchInputs[s].size() : layers[l - 1].neurons.size();
                const vector<double>& layerInput = activations[l]; // input to layer l
                for (int i = 0; i < numNeurons; i++) {
                    for (int j = 0; j < inputSize; j++) {
                        gradW[l][i][j] += deltas[l][i] * layerInput[j];
                    }
                    gradB[l][i] += deltas[l][i];
                }
            }
        } // end for each sample

        // --- Update parameters: average gradients over batch and use optimizer ---
        for (int l = 0; l < L; l++) {
            int numNeurons = layers[l].neurons.size();
            int inputSize = (l == 0) ? batchInputs[0].size() : layers[l - 1].neurons.size();
            for (int i = 0; i < numNeurons; i++) {
                vector<double> avgGrad(inputSize, 0.0);
                for (int j = 0; j < inputSize; j++) {
                    avgGrad[j] = gradW[l][i][j] / batchSize;
                }
                double avgGradBias = gradB[l][i] / batchSize;
                optimizer->updateVector(layers[l].neurons[i].weights,
                                          layers[l].neurons[i].m_w,
                                          layers[l].neurons[i].v_w,
                                          avgGrad);
                optimizer->updateScalar(layers[l].neurons[i].bias,
                                        layers[l].neurons[i].m_b,
                                        layers[l].neurons[i].v_b,
                                        avgGradBias);
            }
        }
    }

    // Fit the network for a given number of epochs using mini-batch training.
    void fit(const vector< vector<double> >& inputs,
             const vector< vector<double> >& targets,
             int epochs, int batchSize = 32)
    {
        int numSamples = inputs.size();
        vector<int> indices(numSamples);
        for (int i = 0; i < numSamples; i++) {
            indices[i] = i;
        }
        random_device rd;
        mt19937 g(rd());

        for (int epoch = 0; epoch < epochs; epoch++) {
            shuffle(indices.begin(), indices.end(), g);
            for (int start = 0; start < numSamples; start += batchSize) {
                int end = min(start + batchSize, numSamples);
                vector< vector<double> > batchInputs, batchTargets;
                for (int i = start; i < end; i++) {
                    batchInputs.push_back(inputs[indices[i]]);
                    batchTargets.push_back(targets[indices[i]]);
                }
                trainBatch(batchInputs, batchTargets);
            }
        }
    }

    // Predict for a single sample.
    vector<double> predict(const vector<double>& input) {
        setTrainingMode(false);
        vector< vector<double> > acts = forwardAllLayers(input);
        return acts.back();
    }

    void setTrainingMode(bool training) {
        isTraining = training;
    }

    void cleanDevice() {
        if (device)
            device->cleanup();
    }
};

//////////////////////////////
// Extern "C" Interface
//////////////////////////////
extern "C" {

NeuralNetwork* createNN(double learningRate, char deviceType) {
    srand((unsigned)time(0));
    return new NeuralNetwork(learningRate, deviceType);
}

void destroyNN(NeuralNetwork* nn) {
    delete nn;
}

// Add the first layer (input size provided)
void addLayerNN(NeuralNetwork* nn, int numNeurons, int inputSize, int activationType) {
    nn->addLayer(numNeurons, inputSize, static_cast<ActivationType>(activationType));
}

// Add a hidden layer (input size inferred from previous layer)
void addHiddenLayerNN(NeuralNetwork* nn, int numNeurons, int activationType) {
    nn->addLayer(numNeurons, static_cast<ActivationType>(activationType));
}

// Add an output layer (same as hidden layer in our simple design)
void addOutputLayerNN(NeuralNetwork* nn, int numNeurons, int activationType) {
    nn->addLayer(numNeurons, static_cast<ActivationType>(activationType));
}

// Fit the network using mini-batch training.
// The inputs and targets are provided as flat arrays.
// 'numSamples' is the number of samples,
// 'inputSize' and 'targetSize' are dimensions per sample,
// 'epochs' is the number of epochs, and 'batchSize' is the mini-batch size.
void fitNN(NeuralNetwork* nn, double* inputs, int numSamples, int inputSize,
           double* targets, int targetSize, int epochs, int batchSize) {
    vector< vector<double> > inVec(numSamples, vector<double>(inputSize));
    vector< vector<double> > tarVec(numSamples, vector<double>(targetSize));
    for (int i = 0; i < numSamples; i++) {
        for (int j = 0; j < inputSize; j++) {
            inVec[i][j] = inputs[i * inputSize + j];
        }
        for (int j = 0; j < targetSize; j++) {
            tarVec[i][j] = targets[i * targetSize + j];
        }
    }
    nn->fit(inVec, tarVec, epochs, batchSize);
}

// Train on a single sample.
void trainSampleNN(NeuralNetwork* nn, double* input, int inputSize, double* target, int targetSize) {
    vector<double> inVec(input, input + inputSize);
    vector<double> tarVec(target, target + targetSize);
    vector< vector<double> > batchInput = { inVec };
    vector< vector<double> > batchTarget = { tarVec };
    nn->trainBatch(batchInput, batchTarget);
}

// Predict for a single sample. The output is written to the provided output array.
void predictNN(NeuralNetwork* nn, double* input, int inputSize, double* output, int outputSize) {
    vector<double> inVec(input, input + inputSize);
    vector<double> pred = nn->predict(inVec);
    int len = min((int)pred.size(), outputSize);
    for (int i = 0; i < len; i++) {
        output[i] = pred[i];
    }
}

// Set training mode (non-zero for true, zero for false).
void setTrainingModeNN(NeuralNetwork* nn, int mode) {
    nn->setTrainingMode(mode != 0);
}

// Clean up device resources.
void cleanDeviceNN(NeuralNetwork* nn) {
    if (nn && nn->device){
        nn->cleanDevice();
        nn->device = nullptr;
        }
}

} // extern "C"
