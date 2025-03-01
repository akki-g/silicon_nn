#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <functional>

using namespace std;


// Activation functions


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
    double activate(vector<double>& inputs, const Activation& act) {
        double sum = 0;
        for (int i = 0; i < weights.size(); i++) {
            sum += inputs[i] * weights[i];
        }
        sum += bias;
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
    vector<double> forward(vector<double>& inputs) {
        vector<double> outputs;
        for (Neuron &neuron : neurons) {
            outputs.push_back(neuron.activate(inputs, activationFunction));
        }

        return outputs;
    }
};



// Neural Network class
class NeuralNetwork {
    public:
    vector<DenseLayer> layers;
    double learningRate;

    NeuralNetwork(double lr = 0.5) : learningRate(lr) {}

    // Add a layer to the network
    // for the first layer, provide inputSize; for the rest it is inferred
    void addLayer(int numNeurons, int inputSize = 0) {
        if (layers.empty()){
            if (inputSize == -1) {
                cerr << "Input size must be provided for the first layer" << endl;
                exit(1);
            }
            layers.push_back(DenseLayer(numNeurons, inputSize));
        }
        else {
            int prevLayerSize = layers.back().neurons.size();
            layers.push_back(DenseLayer(numNeurons, prevLayerSize));
        }
    }

    // Forward pass
    // Takes in a vector of inputs and returns a vector of outputs
    vector<double> forward(vector<double>& inputs) {
        vector<double> outputs = inputs;
        for (DenseLayer &layer : layers) {
            outputs = layer.forward(outputs);
        }
        return outputs;
    }

    // Train the network on one training example using backpropagation
    void train(const vector<double>& input, const vector<double>& target) {
        // 1. Forward pass: stores inputs to each layer (including the initial input)
        vector<vector<double>> layerInputs;
        vector<double> currentInput = input;
        layerInputs.push_back(currentInput);
        for (DenseLayer &layer : layers) {
            currentInput = layer.forward(currentInput);
            layerInputs.push_back(currentInput);
        }
        vector<double> output = currentInput;

        // 2. Backpropagation
        // errors[l] will store the error of each neuron in layer l
        vector<vector<double>> errors(layers.size());

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
};


