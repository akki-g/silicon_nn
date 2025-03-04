import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from neural_network.nn import NeuralNetwork  # Custom Neural Network

# -------------------------
# Data Preparation
# -------------------------
# Load iris dataset.
iris = load_iris()
X = iris.data
y = iris.target

# For binary classification, filter out only the first two classes.
mask = y < 2
X = X[mask]
y = y[mask]

# Standardize the features.
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Reshape targets to be a column vector (and keep values as 0 or 1).
y = y.reshape(-1, 1)

# Split data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Number of epochs
epochs = 10000

# -------------------------
# Benchmark: Custom Neural Network
# -------------------------
print("Benchmarking Custom Neural Network:")
start_time = time.time()
custom_nn = NeuralNetwork(learning_rate=0.5)
custom_nn.add_layer(5, input_size=X_train.shape[1], activation_type=1)  # Hidden layer: ReLU
custom_nn.add_hidden_layer(1, activation_type=0)  # Output layer: Sigmoid
custom_nn.fit(X_train, y_train, epochs=epochs)
custom_time = time.time() - start_time

custom_predictions = [1 if custom_nn.predict(sample)[0] > 0.5 else 0 for sample in X_test]
custom_accuracy = accuracy_score(y_test.flatten(), custom_predictions)
print(f"Custom NN Training Time: {custom_time:.2f} seconds")
print(f"Custom NN Test Accuracy: {custom_accuracy:.2f}")

# -------------------------
# Benchmark: scikit-learn MLPClassifier
# -------------------------
print("\nBenchmarking scikit-learn MLPClassifier:")
start_time = time.time()
mlp = MLPClassifier(hidden_layer_sizes=(5,), activation='relu', solver='adam', max_iter=epochs, random_state=42)
mlp.fit(X_train, y_train.ravel())
sk_time = time.time() - start_time

sk_predictions = mlp.predict(X_test)
sk_accuracy = accuracy_score(y_test.flatten(), sk_predictions)
print(f"MLPClassifier Training Time: {sk_time:.2f} seconds")
print(f"MLPClassifier Test Accuracy: {sk_accuracy:.2f}")

# -------------------------
# Benchmark: TensorFlow (Keras)
# -------------------------
print("\nBenchmarking TensorFlow (Keras):")
start_time = time.time()
tf_model = keras.Sequential([
    keras.layers.Input(shape=(X_train.shape[1],)),  # Explicit Input layer
    keras.layers.Dense(5, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

tf_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
tf_model.fit(X_train, y_train, epochs=epochs, verbose=0, batch_size=10)
tf_time = time.time() - start_time

tf_predictions = (tf_model.predict(X_test) > 0.5).astype(int).flatten()
tf_accuracy = accuracy_score(y_test.flatten(), tf_predictions)
print(f"TensorFlow Training Time: {tf_time:.2f} seconds")
print(f"TensorFlow Test Accuracy: {tf_accuracy:.2f}")

# -------------------------
# Benchmark: PyTorch
# -------------------------
print("\nBenchmarking PyTorch:")
class PyTorchNN(nn.Module):
    def __init__(self):
        super(PyTorchNN, self).__init__()
        self.hidden = nn.Linear(X_train.shape[1], 5)
        self.output = nn.Linear(5, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.hidden(x))
        x = self.sigmoid(self.output(x))
        return x

# Convert data to PyTorch tensors
X_train_torch = torch.tensor(X_train, dtype=torch.float32)
y_train_torch = torch.tensor(y_train, dtype=torch.float32)
X_test_torch = torch.tensor(X_test, dtype=torch.float32)
y_test_torch = torch.tensor(y_test, dtype=torch.float32)

# Initialize model, loss function, and optimizer
pytorch_model = PyTorchNN()
criterion = nn.BCELoss()
optimizer = optim.Adam(pytorch_model.parameters(), lr=0.01)

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
    pytorch_predictions = (pytorch_model(X_test_torch) > 0.5).numpy().astype(int).flatten()
pytorch_accuracy = accuracy_score(y_test.flatten(), pytorch_predictions)
print(f"PyTorch Training Time: {pytorch_time:.2f} seconds")
print(f"PyTorch Test Accuracy: {pytorch_accuracy:.2f}")
