#include "common/Device.h"
#include "cpu/CPUDevice.h"
#include "metal/MetalDevice.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <functional>

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

// ReLU activation function and its derivative.
double relu(double x) {
    return x > 0 ? x : 0.0;
}
double reluDerivative(double activatedValue) {
    return activatedValue > 0 ? 1.0 : 0.0;
}

// Tanh activation function and its derivative.
double tanhActivation(double x) {
    return tanh(x);
}
double tanhDerivative(double activatedValue) {
    return 1.0 - activatedValue * activatedValue;
}


Activation getActivation(ActivationType type) {
    switch (type) {
        case SIGMOID:
            return {sigmoid, sigmoidDerivative};
        case RELU:
            return {relu, reluDerivative};
        case TANH:
            return {tanhActivation, tanhDerivative};
        default:
            cerr << "Invalid activation function type" << endl;
            exit(1);
    }
}

// Neuron class
class Neuron {
    public: 
    vector<double> weights;
    double bias;
    double output;
    double error;

    Neuron(int numInputs) {
        for (int i = 0; i < numInputs; i++) {
            double w = ((double) rand() / RAND_MAX) * 2 - 1;
            weights.push_back(w);
        }
        bias = ((double) rand() / RAND_MAX) * 2 - 1;
        output = 0.0;
    }

    // Activation function
    // Takes in a vector of inputs and returns the output of the neuron
    double activate(vector<double>& inputs, const Activation& act, Device* device) {
        double sum = device->dot(inputs, weights) + bias;
        output = act.func(sum);
        return output;
    }

};



// Dense Layer class
class DenseLayer {
    public:
    vector<Neuron> neurons;
    Activation activationFunction;

    DenseLayer(int numNeurons, int numInputs, const Activation& act = {sigmoid, sigmoidDerivative}) : activationFunction(act) {
        for (int i = 0; i < numNeurons; i++) {
            Neuron n(numInputs);
            neurons.push_back(n);
        }
    }

    // Forward pass
    // Takes in a vector of inputs and returns a vector of outputs
    vector<double> forward(const vector<double>& inputs, Device* device) {
        vector<double> outputs;
        outputs.resize(neurons.size()); 
        for (size_t i = 0; i < neurons.size(); i++) {
            outputs[i] = neurons[i].activate(const_cast<vector<double>&>(inputs), activationFunction, device);
        }
        return outputs;
    }
};



// Neural Network class
class NeuralNetwork {
    public:
    vector<DenseLayer> layers;
    double learningRate;
    Device* device;

    NeuralNetwork(double lr = 0.5, char c = 'c') : learningRate(lr) {
        if (c == 'c') {
            device = new CPUDevice();
        }
        else {
            if (MetalDevice::metalIsAvailable()) {
                cout << "Using device: Metal" << endl;
                device = new MetalDevice();
            }
            else {
                cout << "Metal is not available defualting to CPU" << endl;
                device = new CPUDevice();
            }
        }
    }

    ~NeuralNetwork() {
        delete device;
    }

    // Add a layer to the network
    // for the first layer, provide inputSize; for the rest it is inferred
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
        else {
            int prevLayerSize = layers.back().neurons.size();
            layers.push_back(DenseLayer(numNeurons, prevLayerSize, act));
        }
    }

    // Forward pass
    // Takes in a vector of inputs and returns a vector of outputs
    vector<double> forward(vector<double>& inputs) {
        vector<double> outputs = inputs;
        for (DenseLayer &layer : layers) {
            outputs = layer.forward(outputs, device);
        }
        return outputs;
    }

    // Train the network on one training example using backpropagation
    void train(const vector<double>& input, const vector<double>& target) {
        // 1. Forward pass: stores inputs to each layer (including the initial input)
        vector<vector<double>> layerInputs(layers.size() + 1);
        layerInputs[0] = input;

        for (size_t l = 0; l < layers.size(); l++) {
            vector<double> out = layers[l].forward(layerInputs[l], device);
            layerInputs[l+1] = out;
        }
        vector<double> output = layerInputs.back();

        // 2. Backpropagation
        // errors[l] will store the error of each neuron in layer l
        vector<vector<double>> errors(layers.size());
        for (size_t l = 0; l < layers.size(); l++) {
            errors[l].resize(layers[l].neurons.size(), 0.0);
        }

        // Calculate the error of the output layer
        int lastLayer = layers.size() - 1;
        errors[lastLayer].resize(layers[lastLayer].neurons.size());
        for (int i = 0; i < layers[lastLayer].neurons.size(); i++) {
            double out = layerInputs.back()[i];
            // Error = (target - output) * f'(output)
            errors[lastLayer][i] = (target[i] - out) * layers[lastLayer].activationFunction.derivative(out);

        }

        // Propagate the errors backwards through the hidden layers
        for (int l = layers.size() - 2; l >= 0; l--) {
            errors[l].resize(layers[l].neurons.size());
            for (size_t i=0; i < layers[l].neurons.size(); i++) {
                double error = 0.0;

                for (size_t j = 0; j < layers[l+1].neurons.size(); j++) {
                    error += errors[l+1][j] * layers[l+1].neurons[j].weights[i];
                }
                double out = layerInputs[l+1][i];
                errors[l][i] = error * layers[l].activationFunction.derivative(out);
            }
        }

        // Update the weights and biases
        for (size_t l = 0; l < layers.size(); l++) {
            vector<double> prevLayerOutput = layerInputs[l];
            for (size_t i = 0; i < layers[l].neurons.size(); i++) {
                Neuron &neuron = layers[l].neurons[i];
                for (size_t j = 0; j < prevLayerOutput.size(); j++) {
                    neuron.weights[j] += learningRate * errors[l][i] * prevLayerOutput[j];
                }
                neuron.bias += learningRate * errors[l][i];
            }
        }
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

void addLayerNN(NeuralNetwork* nn, int numNeurons, int inputSize, int activationType) {
    Activation act = getActivation(static_cast<ActivationType>(activationType));
    nn->addLayer(numNeurons, inputSize, act);
}

void addLayerNN_noInput(NeuralNetwork* nn, int numNeurons, int activationType) {
    Activation act = getActivation(static_cast<ActivationType>(activationType));
    nn->addLayer(numNeurons, act);
}

void trainSampleNN(NeuralNetwork* nn, double* input, int inputSize, double* target, int targetSize) {
    vector<double> inputVec(input, input+inputSize);
    vector<double> targetVec(target, target+targetSize);
    nn->train(inputVec, targetVec);
}

void fitNN(NeuralNetwork* nn, double* inputs, int numSample, int inputSize, double* targets, int targetSize, int epochs) {
    for(int epoch = 0; epoch < epochs; epoch++) {
        for(int i = 0; i < numSample; i++) {
            double* inputStart = inputs + i * inputSize;
            double* targetStart = targets + i * targetSize;
            vector<double> inputVec(inputStart, inputStart + inputSize);
            vector<double> targetVec(targetStart, targetStart + targetSize);
            nn->train(inputVec, targetVec);
        }
    }
}

void predictNN(NeuralNetwork* nn, double* input, int inputSize, double* output, int outputSize) {
    vector<double> inputVec(input, input + inputSize);
    vector<double> result = nn->forward(inputVec);
    int len = result.size();
    if (len > outputSize) len = outputSize;
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

    double accuracy = (double) (TP + TN) / (TP + TN + FP + FN);
    double precision = (TP + FP > 0) ? (double) TP / (TP + FP) : 0.0;
    double recall = (TP + FN > 0) ? (double) TP / (TP + FN) : 0.0;
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
