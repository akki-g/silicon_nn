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
#include <Eigen/Dense>
using Eigen::MatrixXd;
using Eigen::VectorXd;

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

double softmax(double x) {
    return exp(x);
}
double softmaxDerivative(double y) {
    return 1.0;
}

// For softmax, we handle it in the forward pass.
// Here we simply return identity.
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
void applyDropout(MatrixXd& A, double prob) {
    for (int i = 0; i < A.rows(); i++) {
        for (int j = 0; j < A.cols(); j++) {
            double r = (double)rand() / RAND_MAX;
            if (r < prob) {
                A(i, j) = 0.0;
            }
            else {
                A(i, j) /= (1.0 - prob);
            }
        }
    }
}

// ---------------------------
// Neuron: stores weights, bias, and Adam accumulators
// ---------------------------
class Neuron {
public:
    vector<double> weights;
    double bias;
    vector<double> m_w, v_w;  // Adam accumulators for weights
    double m_b, v_b;          // Adam accumulators for bias
    double output;            // Latest output after activation

    Neuron(int numInputs) {
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

    // Per-sample forward pass (unchanged)
    vector<double> forward(const vector<double>& inputs, Device* device) {
        size_t numNeurons = neurons.size();
        vector<double> outputs(numNeurons, 0.0);
        if (actType == SOFTMAX) {
            vector<double> logits(numNeurons, 0.0);
            for (size_t i = 0; i < numNeurons; i++) {
                logits[i] = device->dot(inputs, neurons[i].weights) + neurons[i].bias;
            }
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

    // Batched forward pass using Eigen.
    // A: (m x n) matrix, where m = number of samples and n = input dimension.
    // Returns: (m x k) matrix, where k = number of neurons.
    MatrixXd forwardBatch(const MatrixXd &A, Device* device) {
        int m = A.rows();   // number of samples
        int n = A.cols();   // input dimension
        int k = neurons.size(); // number of neurons
        
        // Build weight matrix W (k x n) and bias vector b.
        MatrixXd W(k, n);
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < n; j++) {
                W(i, j) = neurons[i].weights[j];
            }
        }
        VectorXd b(k);
        for (int i = 0; i < k; i++) {
            b(i) = neurons[i].bias;
        }
        
        // Use GPU-accelerated matmul if available.
        MatrixXd Z;
        // dynamic_cast returns non-null if device is a MetalDevice.
        if (dynamic_cast<MetalDevice*>(device)) {
            MetalDevice* gpu = dynamic_cast<MetalDevice*>(device);
            Z = gpu->matmulGPU(A, W, b);
        } else {
            // CPU fallback using Eigen.
            Z = A * W.transpose();
            Z.rowwise() += b.transpose();
        }
        
        MatrixXd A_out;
        if (actType != SOFTMAX) {
            A_out = Z.unaryExpr([this](double z) { return activation.func(z); });
        } else {
            A_out.resize(m, k);
            for (int i = 0; i < m; i++) {
                double maxVal = Z.row(i).maxCoeff();
                VectorXd exps = (Z.row(i).array() - maxVal).exp();
                double sumExps = exps.sum();
                A_out.row(i) = exps.transpose() / sumExps;
            }
            for (int i = 0; i < k; i++) {
                neurons[i].output = A_out.col(i).mean();
            }
        }
        return A_out;
    }
};

// ---------------------------
// NeuralNetwork: main class implementing forward and backpropagation
// ---------------------------
class NeuralNetwork {
private:
    ThreadPool pool;
public:
    vector<DenseLayer> layers;
    Device* device;
    Optimizer* optimizer;
    double learningRate;
    bool useDropout;
    double dropoutProb;
    bool isTraining;

    NeuralNetwork(double lr = 0.001, char devType = 'c', bool dropout = true, double dropProb = 0.1, double weightDecay = 0.0001)
        : learningRate(lr), useDropout(dropout), dropoutProb(dropProb), isTraining(true), pool(std::thread::hardware_concurrency())
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

    // Per-sample forward pass (for compatibility)
    void forwardAllLayers(const vector<double>& input, vector< vector<double> >& activations) {
        int numLayers = layers.size(); 
        activations.resize(numLayers + 1);

        activations[0] = input;

        vector<double> current = input;
        for (size_t l = 0; l < layers.size(); l++) {
            int layerSize = layers[l].neurons.size();
            vector <double> layerOutput(layerSize, 0.0);
            layerOutput = layers[l].forward(current, device);
            if (isTraining && useDropout && l < layers.size() - 1) {
                for (size_t i = 0; i < layerOutput.size(); i++) {
                    double r = (double)rand() / RAND_MAX;
                    if (r < dropoutProb)
                    layerOutput[i] = 0.0;
                    else
                    layerOutput[i] /= (1.0 - dropoutProb);
                }
            }
            activations[l + 1] = layerOutput;
            current = activations[l+1];
        }
    }

    // Batched forward pass using Eigen.
    // Converts a batch of samples (vector<vector<double>>) into an Eigen matrix,
    // then propagates through all layers.
    MatrixXd forwardBatch(const vector< vector<double> > &batchInputs) {
        int m = batchInputs.size();
        int n = batchInputs[0].size();
        MatrixXd X(m, n);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                X(i, j) = batchInputs[i][j];
            }
        }
        MatrixXd A = X;
        for (size_t l = 0; l < layers.size(); l++) {
            A = layers[l].forwardBatch(A, device);
            if (isTraining && useDropout && l < layers.size() - 1) {
                applyDropout(A, dropoutProb);
            }
        }
        return A;
    }

    // Mini-batch training using per-sample backpropagation.
    void trainBatch(const vector< vector<double> >& batchInputs,
        const vector< vector<double> >& batchTargets)
    {
        size_t batchSize = batchInputs.size();
        int L = layers.size();

        // Containers to hold per-sample gradients.
        // gradW_all[s][l][i][j] is the gradient for the j-th weight of neuron i in layer l for sample s.
        vector< vector< vector< vector<double> > > > gradW_all(batchSize);
        // gradB_all[s][l][i] is the gradient for the bias of neuron i in layer l for sample s.
        vector< vector< vector<double> > > gradB_all(batchSize);


        // Process each sample in parallel.
        int numTasks = pool.getWorkerCount();
        int blockSize = (batchSize + numTasks - 1) / numTasks;

        parallel_for<size_t>(0, numTasks, [&](size_t taskIndex) {
        // Compute forward pass for sample s using the per-sample forwardAllLayers method.
            int start = taskIndex * blockSize;
            int end = std::min(start + blockSize, (int)batchSize);
            for (size_t s = start; s < end; s++) {
                vector< vector<double> > activations;
                forwardAllLayers(batchInputs[s], activations);

                // Initialize a container for the deltas for each layer.
                vector< vector<double> > deltas(L);

                // -----------------------
                // Compute delta for the output layer.
                // -----------------------
                int outLayer = L - 1;
                int outSize = activations.back().size();
                deltas[outLayer].resize(outSize, 0.0);
                for (int i = 0; i < outSize; i++) {
                    double a = activations.back()[i];
                    double t = batchTargets[s][i];
                    // For cross-entropy loss with sigmoid/softmax output, delta = (a - t)
                    deltas[outLayer][i] = a - t;
                }

                // -----------------------
                // Backpropagate through hidden layers.
                // -----------------------
                for (int l = L - 2; l >= 0; l--) {
                    int layerSize = layers[l].neurons.size();
                    deltas[l].resize(layerSize, 0.0);
                    for (int i = 0; i < layerSize; i++) {
                        double errorSum = 0.0;
                        int nextLayerNeurons = layers[l + 1].neurons.size();
                        for (int j = 0; j < nextLayerNeurons; j++) {
                            errorSum += layers[l + 1].neurons[j].weights[i] * deltas[l + 1][j];
                        }
                        double a = activations[l + 1][i];
                        deltas[l][i] = errorSum * layers[l].activation.derivative(a);
                    }
                }

                // -----------------------
                // Compute gradients for this sample.
                // -----------------------
                vector< vector< vector<double> > > sampleGradW(L);
                vector< vector<double> > sampleGradB(L);
                for (int l = 0; l < L; l++) {
                    int numNeurons = layers[l].neurons.size();
                    int inputSize = (l == 0) ? batchInputs[s].size() : layers[l - 1].neurons.size();
                    sampleGradW[l].resize(numNeurons, vector<double>(inputSize, 0.0));
                    sampleGradB[l].resize(numNeurons, 0.0);
                    const vector<double>& layerInput = activations[l];
                    for (int i = 0; i < numNeurons; i++) {
                        for (int j = 0; j < inputSize; j++) {
                            sampleGradW[l][i][j] = deltas[l][i] * layerInput[j];
                        }
                        sampleGradB[l][i] = deltas[l][i];
                    }
                }
                gradW_all[s] = sampleGradW;
                gradB_all[s] = sampleGradB;
            }
        }, pool);

        // -----------------------
        // Reduce gradients over all samples and update parameters.
        // -----------------------
        for (int l = 0; l < L; l++) {
            int numNeurons = layers[l].neurons.size();
            int inputSize = (l == 0) ? batchInputs[0].size() : layers[l - 1].neurons.size();
            for (int i = 0; i < numNeurons; i++) {
                vector<double> sumGrad(inputSize, 0.0);
                double sumGradBias = 0.0;
                for (size_t s = 0; s < batchSize; s++) {
                    for (int j = 0; j < inputSize; j++) {
                        sumGrad[j] += gradW_all[s][l][i][j];
                    }
                    sumGradBias += gradB_all[s][l][i];
                }
                // Average the gradients over the batch.
                for (int j = 0; j < inputSize; j++) {
                    sumGrad[j] /= batchSize;
                }
                sumGradBias /= batchSize;
                
                // Update the weights and bias using your optimizer.
                optimizer->updateVector(layers[l].neurons[i].weights,
                                        layers[l].neurons[i].m_w,
                                        layers[l].neurons[i].v_w,
                                        sumGrad);
                optimizer->updateScalar(layers[l].neurons[i].bias,
                                        layers[l].neurons[i].m_b,
                                        layers[l].neurons[i].v_b,
                                        sumGradBias);
            }
        }
    }

    // Fit the network using mini-batch training for a given number of epochs.
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
        vector< vector<double> > acts;
        forwardAllLayers(input, acts);
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

void addLayerNN(NeuralNetwork* nn, int numNeurons, int inputSize, int activationType) {
    nn->addLayer(numNeurons, inputSize, static_cast<ActivationType>(activationType));
}

void addHiddenLayerNN(NeuralNetwork* nn, int numNeurons, int activationType) {
    nn->addLayer(numNeurons, static_cast<ActivationType>(activationType));
}

void addOutputLayerNN(NeuralNetwork* nn, int numNeurons, int activationType) {
    nn->addLayer(numNeurons, static_cast<ActivationType>(activationType));
}

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

void trainSampleNN(NeuralNetwork* nn, double* input, int inputSize, double* target, int targetSize) {
    vector<double> inVec(input, input + inputSize);
    vector<double> tarVec(target, target + targetSize);
    vector< vector<double> > batchInput = { inVec };
    vector< vector<double> > batchTarget = { tarVec };
    nn->trainBatch(batchInput, batchTarget);
}

void predictNN(NeuralNetwork* nn, double* input, int inputSize, double* output, int outputSize) {
    vector<double> inVec(input, input + inputSize);
    vector<double> pred = nn->predict(inVec);
    int len = min((int)pred.size(), outputSize);
    for (int i = 0; i < len; i++) {
        output[i] = pred[i];
    }
}

void setTrainingModeNN(NeuralNetwork* nn, int mode) {
    nn->setTrainingMode(mode != 0);
}

void cleanDeviceNN(NeuralNetwork* nn) {
    if (nn && nn->device){
        nn->cleanDevice();
        nn->device = nullptr;
    }
}

} // extern "C"


