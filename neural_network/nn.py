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

# Activation types
SIGMOID = 0
RELU = 1
TANH = 2
SOFTMAX = 3
LEAKY_RELU = 4
ELU = 5

# Set up function prototypes to match the C++ exports

# createNN: takes a double (learning rate) and a char (device type) and returns a pointer.
_lib.createNN.restype = ctypes.c_void_p
_lib.createNN.argtypes = [ctypes.c_double, ctypes.c_char]

# destroyNN: takes a pointer and returns void
_lib.destroyNN.restype = None
_lib.destroyNN.argtypes = [ctypes.c_void_p]

# addLayerNN: used for the first layer (with input size)
_lib.addLayerNN.restype = None
_lib.addLayerNN.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]

# Add hidden and output layers
_lib.addHiddenLayerNN.restype = None
_lib.addHiddenLayerNN.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]

_lib.addOutputLayerNN.restype = None
_lib.addOutputLayerNN.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]

# Add layers with batch normalization
_lib.addBatchNormLayerNN.restype = None
_lib.addBatchNormLayerNN.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]

# Add layers with dropout
_lib.addDropoutLayerNN.restype = None
_lib.addDropoutLayerNN.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_double]

# fitNN: train the network on a dataset
_lib.fitNN.restype = None
_lib.fitNN.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int, ctypes.c_int
]

# predictNN: predict output for a single sample
_lib.predictNN.restype = None
_lib.predictNN.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_double), ctypes.c_int,
    ctypes.POINTER(ctypes.c_double), ctypes.c_int
]

# setTrainingModeNN: set training mode
_lib.setTrainingModeNN.restype = None
_lib.setTrainingModeNN.argtypes = [ctypes.c_void_p, ctypes.c_int]

# cleanDeviceNN: clean up device resources
_lib.cleanDeviceNN.restype = None
_lib.cleanDeviceNN.argtypes = [ctypes.c_void_p]

class NeuralNetwork:
    """
    Python wrapper for the optimized C++ neural network library.
    This class provides a high-level interface to create, train, and use
    neural networks accelerated with CPU or GPU computation.
    """
    def __init__(self, learning_rate=0.001, device='c', weight_decay=0.0001):
        """
        Initialize a new neural network.
        
        Args:
            learning_rate (float): Learning rate for training
            device (str): Device type - 'c' for CPU, 'g' for GPU (Metal on macOS)
            weight_decay (float): L2 regularization strength
        """
        self.nn_ptr = _lib.createNN(
            ctypes.c_double(learning_rate),
            ctypes.c_char(device.encode())
        )
        self.output_size = None

    def add_layer(self, num_neurons, input_size, activation_type=RELU):
        """
        Add the first layer with specified input size.
        
        Args:
            num_neurons (int): Number of neurons in the layer
            input_size (int): Dimension of input data
            activation_type (int): Activation function type
                (SIGMOID=0, RELU=1, TANH=2, SOFTMAX=3, LEAKY_RELU=4, ELU=5)
        """
        _lib.addLayerNN(
            self.nn_ptr,
            ctypes.c_int(num_neurons),
            ctypes.c_int(input_size),
            ctypes.c_int(activation_type)
        )
        self.output_size = num_neurons
    
    def add_hidden_layer(self, num_neurons, activation_type=RELU):
        """
        Add a hidden layer.
        
        Args:
            num_neurons (int): Number of neurons in the layer
            activation_type (int): Activation function type
        """
        _lib.addHiddenLayerNN(
            self.nn_ptr,
            ctypes.c_int(num_neurons),
            ctypes.c_int(activation_type)
        )
        self.output_size = num_neurons

    def add_output_layer(self, num_neurons, activation_type=SIGMOID):
        """
        Add the output layer.
        
        Args:
            num_neurons (int): Number of neurons in the layer
            activation_type (int): Activation function type
        """
        _lib.addOutputLayerNN(
            self.nn_ptr,
            ctypes.c_int(num_neurons),
            ctypes.c_int(activation_type)
        )
        self.output_size = num_neurons
    
    def add_batch_norm_layer(self, num_neurons, activation_type=RELU):
        """
        Add a layer with batch normalization.
        
        Args:
            num_neurons (int): Number of neurons in the layer
            activation_type (int): Activation function type
        """
        _lib.addBatchNormLayerNN(
            self.nn_ptr,
            ctypes.c_int(num_neurons),
            ctypes.c_int(activation_type)
        )
        self.output_size = num_neurons
    
    def add_dropout_layer(self, num_neurons, activation_type=RELU, dropout_rate=0.5):
        """
        Add a layer with dropout.
        
        Args:
            num_neurons (int): Number of neurons in the layer
            activation_type (int): Activation function type
            dropout_rate (float): Probability of dropping a neuron (0.0 to 1.0)
        """
        _lib.addDropoutLayerNN(
            self.nn_ptr,
            ctypes.c_int(num_neurons),
            ctypes.c_int(activation_type),
            ctypes.c_double(dropout_rate)
        )
        self.output_size = num_neurons

    def fit(self, inputs, targets, epochs, batch_size=32):
        """
        Train the network on a dataset.
        
        Args:
            inputs (numpy.ndarray): Input data of shape (num_samples, input_size)
            targets (numpy.ndarray): Target data of shape (num_samples, output_size)
            epochs (int): Number of training epochs
            batch_size (int): Mini-batch size
        """
        # Ensure inputs and targets are numpy arrays
        inputs = np.asarray(inputs, dtype=np.float64)
        targets = np.asarray(targets, dtype=np.float64)
        
        num_samples, input_size = inputs.shape
        if len(targets.shape) == 1:
            targets = targets.reshape(-1, 1)
        _, output_size = targets.shape

        # Flatten arrays for C++ interface
        inputs_flat = np.ascontiguousarray(inputs.flatten(), dtype=np.float64)
        targets_flat = np.ascontiguousarray(targets.flatten(), dtype=np.float64)

        # Get pointers to data
        inputs_ptr = inputs_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        targets_ptr = targets_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        # Call C++ function
        _lib.fitNN(
            self.nn_ptr,
            inputs_ptr,
            ctypes.c_int(num_samples),
            ctypes.c_int(input_size),
            targets_ptr,
            ctypes.c_int(output_size),
            ctypes.c_int(epochs),
            ctypes.c_int(batch_size)
        )
    
    def predict(self, input_data):
        """
        Predict output for input data.
        
        Args:
            input_data (numpy.ndarray): Input data
                If 1D array: single sample
                If 2D array: multiple samples
        
        Returns:
            numpy.ndarray: Predicted outputs
        """
        # Set to evaluation mode
        self.set_training_mode(False)
        
        # Handle both single sample and batch prediction
        single_sample = False
        if len(input_data.shape) == 1:
            input_data = input_data.reshape(1, -1)
            single_sample = True
        
        num_samples, input_size = input_data.shape
        
        # If output_size is not set, use 1 as default
        if self.output_size is None:
            self.output_size = 1
        
        # Create output array
        predictions = np.zeros((num_samples, self.output_size), dtype=np.float64)
        
        # Make predictions for each sample
        for i in range(num_samples):
            sample = np.ascontiguousarray(input_data[i], dtype=np.float64)
            output = np.zeros(self.output_size, dtype=np.float64)
            
            # Get pointers to data
            sample_ptr = sample.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            output_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            
            # Call C++ function
            _lib.predictNN(
                self.nn_ptr,
                sample_ptr,
                ctypes.c_int(input_size),
                output_ptr,
                ctypes.c_int(self.output_size)
            )
            
            predictions[i] = output
        
        # Return single sample or batch predictions
        if single_sample:
            return predictions[0]
        else:
            return predictions
    
    def predict_classes(self, input_data, threshold=0.5):
        """
        Predict class labels for input data.
        
        Args:
            input_data (numpy.ndarray): Input data
            threshold (float): Classification threshold for binary classification
        
        Returns:
            numpy.ndarray: Predicted class labels
        """
        probs = self.predict(input_data)
        
        # Binary classification
        if probs.shape[1] == 1:
            return (probs > threshold).astype(np.int32)
        
        # Multi-class classification
        else:
            return np.argmax(probs, axis=1)
    
    def set_training_mode(self, is_training):
        """
        Set training mode (affects dropout, batch norm, etc.)
        
        Args:
            is_training (bool): Whether the network is in training mode
        """
        _lib.setTrainingModeNN(
            self.nn_ptr,
            ctypes.c_int(1 if is_training else 0)
        )
    
    def clean_device(self):
        """Clean up device resources."""
        _lib.cleanDeviceNN(self.nn_ptr)
    
    def __del__(self):
        """Destructor to clean up C++ resources."""
        if hasattr(self, 'nn_ptr') and self.nn_ptr:
            self.clean_device()
            _lib.destroyNN(self.nn_ptr)
            self.nn_ptr = None