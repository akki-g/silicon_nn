import os 
import platform
import ctypes 
import numpy as np

if platform.system() == "Darwin":
    lib_name = "libnn.dylib"
else:
    lib_name = "libnn.so"

lib_path = os.path.join(os.path.dirname(__file__), lib_name)
_lib = ctypes.CDLL(lib_path)

_lib.createNN.restype = ctypes.c_void_p
_lib.createNN.argtypes = [ctypes]

_lib.addLayerNN.restype = None
_lib.addLayerNN.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]

_lib.addLayerNN_noInput.restype = None
_lib.addLayerNN_noInput.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]

_lib.trainNN.restype = None
_lib.trainNN.argtypes = [ctypes.c_void_p,
                       ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int,
                       ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int]

_lib.predictNN.restype = None
_lib.predictNN.argtypes = [ctypes.c_void_p,
                       ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int,
                       ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int,
                       ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                       ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]

_lib.destroyNN.restype = None
_lib.destroyNN.argtypes = [ctypes.c_void_p]

class NeuralNetwork:

    def __init__(self, learning_rate=0.5):
        self.nn_ptr = _lib.createNN(ctypes.c_double(learning_rate))

    def add_layer(self, num_neurons, input_size, activation_type):
        _lib.addLayerNN(self.nn_ptr, ctypes.c_int(num_neurons), ctypes.c_int(input_size), ctypes.c_int(activation_type))
    
    def add_hidden_layer(self, num_neurons, activation_type):
        _lib.addLayerNN_noInput(self.nn_ptr, ctypes.c_int(num_neurons), ctypes.c_int(activation_type))

    def fit(self, inputs, targets, epochs):
        num_samples, input_size = inputs.shape
        _, output_size = targets.shape

        inputs_flat = inputs.flatten().astype(ctypes.c_double)
        targets_flat = targets.flatten().astype(ctypes.c_double)

        inputs_ptr = inputs_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        targets_ptr = targets_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        _lib.fitNN(self.nn_ptr, inputs_ptr, ctypes.c_int(num_samples), ctypes.c_int(input_size), targets_ptr, ctypes.c_int(output_size), ctypes.c_int(epochs))

    def predict(self, inputs):

        input_size = inputs.shape[0]
        input_data = inputs.astype(ctypes.c_double)
        input_ptr = input_data.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        output_arr = np.zeros((input_size, 1))
        output_ptr = output_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        _lib.predictNN(self.nn_ptr, input_ptr, ctypes.c_int(input_size), output_ptr, ctypes.c_int(1))

        return output_arr
    

    def evaluate(self, inputs, targets):

        num_samples, input_size = inputs.shape
        _, output_size = targets.shape

        inputs_flat = inputs.flatten().astype(ctypes.c_double)
        targets_flat = targets.flatten().astype(ctypes.c_double)

        inputs_ptr = inputs_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        targets_ptr = targets_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        accuracy = ctypes.c_double()
        precision = ctypes.c_double()
        recall = ctypes.c_double()
        f1 = ctypes.c_double()

        _lib.evaluateNN(self.nn_ptr, inputs_ptr, ctypes.c_int(num_samples), ctypes.c_int(input_size), targets_ptr, ctypes.c_int(output_size), ctypes.byref(accuracy), ctypes.byref(precision), ctypes.byref(recall), ctypes.byref(f1))

        return {
            "accuracy": accuracy.value,
            "precision": precision.value,
            "recall": recall.value,
            "f1": f1.value
        }
    
    def __del__(self):
        if self.nn_ptr:
            _lib.destroyNN(self.nn_ptr)
            self.nn_ptr = None