import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from neural_network.nn import NeuralNetwork  # Custom Neural Network

# -------------------------
# Data Preparation (MNIST)
# -------------------------
# Load MNIST dataset from TensorFlow
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Normalize pixel values (0-255 → 0-1)
X_train, X_test = X_train / 255.0, X_test / 255.0

# Flatten images (28x28 → 784 features)
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Standardize data
scaler = StandardScaler()
X_train_flat = scaler.fit_transform(X_train_flat)
X_test_flat = scaler.transform(X_test_flat)

# Use only 2 classes (digits 0 and 1) for binary classification (like original Iris example)
mask_train = (y_train == 0) | (y_train == 1)
mask_test = (y_test == 0) | (y_test == 1)

X_train_flat, y_train = X_train_flat[mask_train], y_train[mask_train]
X_test_flat, y_test = X_test_flat[mask_test], y_test[mask_test]

# Reshape targets to be column vectors
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# Number of epochs
epochs = 100  # Reduced to 100 to avoid excessive runtime

# -------------------------
# Benchmark: Custom Neural Network
# -------------------------

print("Benchmarking Custom Neural Network:")
start_time = time.time()
lr = 0.01
custom_nn = NeuralNetwork(lr, "c")
custom_nn.add_layer(128, input_size=X_train_flat.shape[1], activation_type=1)  # Hidden layer: ReLU
custom_nn.add_hidden_layer(128, activation_type=1)  # Hidden layer: ReLU
custom_nn.add_hidden_layer(128, activation_type=1)  # Hidden layer: ReLU
custom_nn.add_hidden_layer(128, activation_type=1)  # Hidden layer: ReLU
custom_nn.add_output_layer(1, activation_type=0)  # Output layer: Sigmoid
custom_nn.fit(X_train_flat, y_train, epochs=epochs)
custom_time = time.time() - start_time

custom_predictions = [1 if custom_nn.predict(sample)[0] > 0.5 else 0 for sample in X_test_flat]
custom_accuracy = accuracy_score(y_test.flatten(), custom_predictions)
print(f"Custom NN Training Time: {custom_time:.2f} seconds")
print(f"Custom NN Test Accuracy: {custom_accuracy:.2f}")
custom_nn.cleanDevice()


# -------------------------
# Benchmark: scikit-learn MLPClassifier
# -------------------------
print("\nBenchmarking scikit-learn MLPClassifier:")
start_time = time.time()
mlp = MLPClassifier(hidden_layer_sizes=(128, 128, 128, 128), activation='relu', solver='sgd', learning_rate_init=0.01, learning_rate='constant', max_iter=epochs, random_state=42)
mlp.fit(X_train_flat, y_train.ravel())
sk_time = time.time() - start_time

sk_predictions = mlp.predict(X_test_flat)
sk_accuracy = accuracy_score(y_test.flatten(), sk_predictions)

print(f"MLPClassifier Training Time: {sk_time:.2f} seconds")
print(f"MLPClassifier Test Accuracy: {sk_accuracy:.2f}")

# -------------------------
# Benchmark: PyTorch (Using Metal MPS)
# -------------------------
print("\nBenchmarking PyTorch:")
device = torch.device("cpu")
print(f"Using device: {device}")

class PyTorchNN(nn.Module):
    def __init__(self):
        super(PyTorchNN, self).__init__()
        self.hidden1 = nn.Linear(X_train_flat.shape[1], 128)
        self.hidden2 = nn.Linear(128, 128)
        self.hidden3 = nn.Linear(128, 128)
        self.hidden4 = nn.Linear(128, 128)
        self.output = nn.Linear(128, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.hidden1(x))
        x = self.relu(self.hidden2(x))
        x = self.relu(self.hidden3(x))
        x = self.relu(self.hidden4(x))
        x = self.sigmoid(self.output(x))
        return x

# Convert data to PyTorch tensors
X_train_torch = torch.tensor(X_train_flat, dtype=torch.float32).to(device)
y_train_torch = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test_torch = torch.tensor(X_test_flat, dtype=torch.float32).to(device)
y_test_torch = torch.tensor(y_test, dtype=torch.float32).to(device)

# Initialize model, loss function, and optimizer
pytorch_model = PyTorchNN().to(device)
optimizer = optim.SGD(pytorch_model.parameters(), lr=0.01, momentum=0.9)
criterion = nn.BCELoss()
start_time = time.time()

# Training loop
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = pytorch_model(X_train_torch)
    loss = criterion(outputs, y_train_torch)
    loss.backward()
    optimizer.step()

pytorch_time = time.time() - start_time

# Evaluation
with torch.no_grad():
    X_test_torch = X_test_torch.to(device)  # Move test data to MPS
    pytorch_predictions = (pytorch_model(X_test_torch) > 0.5).cpu().numpy().astype(int).flatten()  # Move predictions to CPU

pytorch_accuracy = accuracy_score(y_test.flatten(), pytorch_predictions)

print(f"PyTorch Training Time: {pytorch_time:.2f} seconds")
print(f"PyTorch Test Accuracy: {pytorch_accuracy:.2f}")
