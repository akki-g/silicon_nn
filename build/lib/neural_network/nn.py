import os
import platform
import ctypes
import numpy as np

# Determine the shared library name based on the platform.
if platform.system() == "Darwin":
    lib_name = "libnn.dylib"
else:
    lib_name = "libnn.so"

# Construct the path to the shared library (assumed to be in the same directory as this file).
lib_path = os.path.join(os.path.dirname(__file__), lib_name)
_lib = ctypes.CDLL(lib_path)

# Set up function prototypes to match the C++ exports.

# createNN: takes a double (learning rate) and returns a pointer.
_lib.createNN.restype = ctypes.c_void_p
_lib.createNN.argtypes = [ctypes.c_double]

# addLayerNN: (pointer, int numNeurons, int inputSize, int activationType) -> void.
_lib.addLayerNN.restype = None
_lib.addLayerNN.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]

# addLayerNN_noInput: (pointer, int numNeurons, int activationType) -> void.
_lib.addLayerNN_noInput.restype = None
_lib.addLayerNN_noInput.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]

# fitNN: (pointer, double* inputs, int numSamples, int inputSize, double* targets, int outputSize, int epochs) -> void.
_lib.fitNN.restype = None
_lib.fitNN.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int
]

# predictNN: (pointer, double* input, int inputSize, double* output, int outputSize) -> void.
_lib.predictNN.restype = None
_lib.predictNN.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_double), ctypes.c_int,
    ctypes.POINTER(ctypes.c_double), ctypes.c_int
]

# destroyNN: (pointer) -> void.
_lib.destroyNN.restype = None
_lib.destroyNN.argtypes = [ctypes.c_void_p]

class NeuralNetwork:
    """Python wrapper for the C++ NeuralNetwork library."""
    def __init__(self, learning_rate=0.5):
        self.nn_ptr = _lib.createNN(ctypes.c_double(learning_rate))

    def add_layer(self, num_neurons, input_size, activation_type):
        """
        Add the first layer. 
        :param num_neurons: Number of neurons.
        :param input_size: Dimension of input.
        :param activation_type: 0 for SIGMOID, 1 for RELU, 2 for TANH.
        """
        _lib.addLayerNN(self.nn_ptr, ctypes.c_int(num_neurons),
                        ctypes.c_int(input_size), ctypes.c_int(activation_type))
    
    def add_hidden_layer(self, num_neurons, activation_type):
        """
        Add a subsequent (hidden or output) layer.
        :param num_neurons: Number of neurons.
        :param activation_type: 0 for SIGMOID, 1 for RELU, 2 for TANH.
        """
        _lib.addLayerNN_noInput(self.nn_ptr, ctypes.c_int(num_neurons), ctypes.c_int(activation_type))
    
    def fit(self, inputs, targets, epochs):
        """
        Train the network on a dataset.
        :param inputs: numpy array of shape (num_samples, input_size)
        :param targets: numpy array of shape (num_samples, output_size)
        :param epochs: Number of training epochs.
        """
        num_samples, input_size = inputs.shape
        _, output_size = targets.shape

        # Ensure the data is of type double.
        inputs_flat = np.ascontiguousarray(inputs.flatten(), dtype=np.double)
        targets_flat = np.ascontiguousarray(targets.flatten(), dtype=np.double)

        inputs_ptr = inputs_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        targets_ptr = targets_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        _lib.fitNN(self.nn_ptr, inputs_ptr, ctypes.c_int(num_samples),
                   ctypes.c_int(input_size), targets_ptr, ctypes.c_int(output_size),
                   ctypes.c_int(epochs))
    
    def predict(self, input_sample):
        """
        Predict the output for a single input sample.
        :param input_sample: numpy array of shape (input_size,)
        :return: numpy array of predictions (of size output_size).
        """
        # Ensure input_sample is contiguous and of type double.
        input_sample = np.ascontiguousarray(input_sample, dtype=np.double)
        input_size = input_sample.shape[0]
        input_ptr = input_sample.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        # Assuming output_size = 1; adjust if needed.
        output_size = 1
        output_arr = np.zeros(output_size, dtype=np.double)
        output_ptr = output_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        _lib.predictNN(self.nn_ptr, input_ptr, ctypes.c_int(input_size),
                       output_ptr, ctypes.c_int(output_size))
        return output_arr

    def __del__(self):
        if self.nn_ptr:
            _lib.destroyNN(self.nn_ptr)
            self.nn_ptr = None
