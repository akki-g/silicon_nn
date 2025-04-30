import os
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import psutil
import gc

# Import neural network frameworks
try:
    from neural_network.nn import NeuralNetwork, SIGMOID, RELU, TANH, SOFTMAX
    silicon_nn_available = True
except ImportError:
    print("silicon_nn not available, skipping")
    silicon_nn_available = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    torch_available = True
except ImportError:
    print("PyTorch not available, skipping")
    torch_available = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    tf_available = True
except ImportError:
    print("TensorFlow not available, skipping")
    tf_available = False

try:
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    sklearn_available = True
except ImportError:
    print("scikit-learn not available, skipping")
    sklearn_available = False

# Set up argument parser
parser = argparse.ArgumentParser(description='Neural Network Performance Benchmark')
parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10'], 
                    help='Dataset to use for benchmarking')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train')
parser.add_argument('--hidden_size', type=int, default=128, help='Size of hidden layers')
parser.add_argument('--num_layers', type=int, default=4, help='Number of hidden layers')
parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'gpu'], 
                    help='Device to use (CPU or GPU)')
parser.add_argument('--output', type=str, default='benchmark_results', help='Output directory for results')
parser.add_argument('--skip_slow', action='store_true', help='Skip slow frameworks (like scikit-learn) for large datasets')

args = parser.parse_args()

# Create output directory if it doesn't exist
os.makedirs(args.output, exist_ok=True)

# Function to load datasets
def load_dataset(dataset_name):
    """Load and preprocess the selected dataset"""
    if dataset_name == 'mnist':
        if tf_available:
            # Use TensorFlow's dataset loading utilities
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
            
            # Reshape and normalize
            x_train = x_train.reshape(-1, 28*28).astype('float32') / 255.0
            x_test = x_test.reshape(-1, 28*28).astype('float32') / 255.0
            
            # One-hot encode labels
            if tf_available:
                y_train_onehot = tf.keras.utils.to_categorical(y_train, 10)
                y_test_onehot = tf.keras.utils.to_categorical(y_test, 10)
            else:
                # Fallback one-hot encoding
                y_train_onehot = np.zeros((y_train.size, 10))
                y_train_onehot[np.arange(y_train.size), y_train] = 1
                y_test_onehot = np.zeros((y_test.size, 10))
                y_test_onehot[np.arange(y_test.size), y_test] = 1
                
            input_dim = 28*28
            output_dim = 10
        else:
            raise ImportError("TensorFlow is required to load MNIST dataset")
            
    elif dataset_name == 'cifar10':
        if tf_available:
            # Load CIFAR-10
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
            
            # Reshape and normalize
            x_train = x_train.reshape(-1, 32*32*3).astype('float32') / 255.0
            x_test = x_test.reshape(-1, 32*32*3).astype('float32') / 255.0
            
            # One-hot encode labels
            y_train = y_train.reshape(-1)
            y_test = y_test.reshape(-1)
            
            if tf_available:
                y_train_onehot = tf.keras.utils.to_categorical(y_train, 10)
                y_test_onehot = tf.keras.utils.to_categorical(y_test, 10)
            else:
                # Fallback one-hot encoding
                y_train_onehot = np.zeros((y_train.size, 10))
                y_train_onehot[np.arange(y_train.size), y_train] = 1
                y_test_onehot = np.zeros((y_test.size, 10))
                y_test_onehot[np.arange(y_test.size), y_test] = 1
                
            input_dim = 32*32*3
            output_dim = 10
        else:
            raise ImportError("TensorFlow is required to load CIFAR-10 dataset")
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    print(f"Dataset: {dataset_name}")
    print(f"Training set: {x_train.shape[0]} samples")
    print(f"Test set: {x_test.shape[0]} samples")
    print(f"Input dimension: {input_dim}")
    print(f"Output dimension: {output_dim}")
    
    # Standardize data
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    
    return x_train, y_train, y_train_onehot, x_test, y_test, y_test_onehot, input_dim, output_dim

def get_memory_usage():
    """Get current memory usage of the process in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def run_benchmark():
    """Run benchmark of all available neural network libraries"""
    results = []
    
    # Load dataset
    x_train, y_train, y_train_onehot, x_test, y_test, y_test_onehot, input_dim, output_dim = load_dataset(args.dataset)
    
    # Create smaller subsets for quicker testing if needed
    if args.dataset == 'cifar10' and args.skip_slow:
        train_subset_size = min(20000, len(x_train))
        test_subset_size = min(5000, len(x_test))
        indices = np.random.choice(len(x_train), train_subset_size, replace=False)
        x_train_small = x_train[indices]
        y_train_small = y_train[indices]
        y_train_onehot_small = y_train_onehot[indices]
        
        indices = np.random.choice(len(x_test), test_subset_size, replace=False)
        x_test_small = x_test[indices]
        y_test_small = y_test[indices]
        y_test_onehot_small = y_test_onehot[indices]
    else:
        x_train_small = x_train
        y_train_small = y_train
        y_train_onehot_small = y_train_onehot
        x_test_small = x_test
        y_test_small = y_test
        y_test_onehot_small = y_test_onehot
    
    # 1. Benchmark silicon_nn
    if silicon_nn_available:
        print("\n======= Testing silicon_nn =======")
        try:
            # Track memory before
            mem_before = get_memory_usage()
            
            # Create model
            device_type = 'g' if args.device == 'gpu' else 'c'
            model = NeuralNetwork(learning_rate=0.001, device=device_type)
            
            # Add layers
            model.add_layer(args.hidden_size, input_size=input_dim, activation_type=RELU)
            for _ in range(args.num_layers - 1):
                model.add_hidden_layer(args.hidden_size, activation_type=RELU)
            model.add_output_layer(output_dim, activation_type=SOFTMAX)
            
            # Training
            start_time = time.time()
            model.fit(x_train_small, y_train_onehot_small, epochs=args.epochs, batch_size=args.batch_size)
            train_time = time.time() - start_time
            
            # Memory usage
            mem_after = get_memory_usage()
            memory_usage = mem_after - mem_before
            
            # Inference
            start_time = time.time()
            predictions = np.array([model.predict(x) for x in x_test_small])
            infer_time = time.time() - start_time
            
            # Calculate accuracy
            pred_labels = np.argmax(predictions, axis=1)
            accuracy = np.mean(pred_labels == y_test_small) * 100.0
            
            print(f"Silicon NN - Training time: {train_time:.2f}s, Inference time: {infer_time:.2f}s")
            print(f"Silicon NN - Memory usage: {memory_usage:.2f} MB")
            print(f"Silicon NN - Accuracy: {accuracy:.2f}%")
            
            results.append({
                'framework': 'silicon_nn',
                'device': args.device,
                'train_time': train_time,
                'infer_time': infer_time,
                'memory_mb': memory_usage,
                'accuracy': accuracy,
                'epochs': args.epochs,
                'batch_size': args.batch_size,
            })
            
            # Clean up
            del model, predictions
            gc.collect()
            
        except Exception as e:
            print(f"Error testing silicon_nn: {e}")
    
    # 2. Benchmark PyTorch
    if torch_available:
        print("\n======= Testing PyTorch =======")
        try:
            # Select device
            device = torch.device("cuda" if args.device == 'gpu' and torch.cuda.is_available() else 
                                 "mps" if args.device == 'gpu' and torch.backends.mps.is_available() else 
                                 "cpu")
            print(f"Using PyTorch device: {device}")
            
            # Define model
            class PyTorchModel(nn.Module):
                def __init__(self):
                    super(PyTorchModel, self).__init__()
                    layers = []
                    layers.append(nn.Linear(input_dim, args.hidden_size))
                    layers.append(nn.ReLU())
                    
                    for _ in range(args.num_layers - 1):
                        layers.append(nn.Linear(args.hidden_size, args.hidden_size))
                        layers.append(nn.ReLU())
                    
                    layers.append(nn.Linear(args.hidden_size, output_dim))
                    self.model = nn.Sequential(*layers)
                
                def forward(self, x):
                    return self.model(x)
            
            # Track memory
            mem_before = get_memory_usage()
            
            # Create model, loss, and optimizer
            model = PyTorchModel().to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # Convert data to PyTorch tensors
            x_train_tensor = torch.tensor(x_train_small, dtype=torch.float32).to(device)
            y_train_tensor = torch.tensor(y_train_small, dtype=torch.long).to(device)
            x_test_tensor = torch.tensor(x_test_small, dtype=torch.float32).to(device)
            
            # Create DataLoader
            train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            
            # Training
            start_time = time.time()
            model.train()
            for epoch in range(args.epochs):
                for inputs, targets in train_loader:
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
            train_time = time.time() - start_time
            
            # Memory usage
            mem_after = get_memory_usage()
            memory_usage = mem_after - mem_before
            
            # Inference
            start_time = time.time()
            model.eval()
            with torch.no_grad():
                predictions = model(x_test_tensor)
            infer_time = time.time() - start_time
            
            # Calculate accuracy
            _, predicted = torch.max(predictions.data, 1)
            accuracy = (predicted.cpu().numpy() == y_test_small).mean() * 100.0
            
            print(f"PyTorch - Training time: {train_time:.2f}s, Inference time: {infer_time:.2f}s")
            print(f"PyTorch - Memory usage: {memory_usage:.2f} MB")
            print(f"PyTorch - Accuracy: {accuracy:.2f}%")
            
            results.append({
                'framework': 'pytorch',
                'device': str(device),
                'train_time': train_time,
                'infer_time': infer_time,
                'memory_mb': memory_usage,
                'accuracy': accuracy,
                'epochs': args.epochs,
                'batch_size': args.batch_size,
            })
            
            # Clean up
            del model, optimizer, criterion, x_train_tensor, y_train_tensor, x_test_tensor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            print(f"Error testing PyTorch: {e}")
    
    # 3. Benchmark TensorFlow
    if tf_available:
        print("\n======= Testing TensorFlow =======")
        try:
            # Configure TensorFlow to use GPU if requested
            physical_devices = tf.config.list_physical_devices('GPU')
            if args.device == 'gpu' and len(physical_devices) > 0:
                tf.config.experimental.set_memory_growth(physical_devices[0], True)
                print(f"Using TensorFlow with GPU: {physical_devices[0]}")
            elif args.device == 'gpu':
                print("GPU requested but not available for TensorFlow")
            else:
                print("Using TensorFlow with CPU")
                # Disable GPU for TensorFlow
                tf.config.set_visible_devices([], 'GPU')
            
            # Track memory
            mem_before = get_memory_usage()
            
            # Create model
            model = Sequential()
            model.add(Dense(args.hidden_size, activation='relu', input_shape=(input_dim,)))
            for _ in range(args.num_layers - 1):
                model.add(Dense(args.hidden_size, activation='relu'))
            model.add(Dense(output_dim, activation='softmax'))
            
            # Compile model
            model.compile(optimizer=Adam(learning_rate=0.001),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
            
            # Training
            start_time = time.time()
            model.fit(x_train_small, y_train_onehot_small, 
                      epochs=args.epochs, 
                      batch_size=args.batch_size,
                      verbose=0)
            train_time = time.time() - start_time
            
            # Memory usage
            mem_after = get_memory_usage()
            memory_usage = mem_after - mem_before
            
            # Inference
            start_time = time.time()
            predictions = model.predict(x_test_small, verbose=0)
            infer_time = time.time() - start_time
            
            # Calculate accuracy
            pred_labels = np.argmax(predictions, axis=1)
            accuracy = np.mean(pred_labels == y_test_small) * 100.0
            
            print(f"TensorFlow - Training time: {train_time:.2f}s, Inference time: {infer_time:.2f}s")
            print(f"TensorFlow - Memory usage: {memory_usage:.2f} MB")
            print(f"TensorFlow - Accuracy: {accuracy:.2f}%")
            
            results.append({
                'framework': 'tensorflow',
                'device': 'gpu' if len(physical_devices) > 0 and args.device == 'gpu' else 'cpu',
                'train_time': train_time,
                'infer_time': infer_time,
                'memory_mb': memory_usage,
                'accuracy': accuracy,
                'epochs': args.epochs,
                'batch_size': args.batch_size,
            })
            
            # Clean up
            del model, predictions
            tf.keras.backend.clear_session()
            gc.collect()
            
        except Exception as e:
            print(f"Error testing TensorFlow: {e}")
    
    # 4. Benchmark scikit-learn (CPU only)
    if sklearn_available and not (args.skip_slow and args.dataset == 'cifar10'):
        print("\n======= Testing scikit-learn MLPClassifier (CPU only) =======")
        try:
            # Track memory
            mem_before = get_memory_usage()
            
            # Create hidden layer sizes tuple
            hidden_layer_sizes = tuple([args.hidden_size] * args.num_layers)
            
            # Create model
            model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                                 activation='relu',
                                 solver='adam',
                                 alpha=0.0001,
                                 batch_size=args.batch_size,
                                 learning_rate_init=0.001,
                                 max_iter=args.epochs,
                                 shuffle=True,
                                 random_state=42,
                                 verbose=False)
            
            # Training
            start_time = time.time()
            model.fit(x_train_small, y_train_small)
            train_time = time.time() - start_time
            
            # Memory usage
            mem_after = get_memory_usage()
            memory_usage = mem_after - mem_before
            
            # Inference
            start_time = time.time()
            predictions = model.predict(x_test_small)
            infer_time = time.time() - start_time
            
            # Calculate accuracy
            accuracy = np.mean(predictions == y_test_small) * 100.0
            
            print(f"scikit-learn - Training time: {train_time:.2f}s, Inference time: {infer_time:.2f}s")
            print(f"scikit-learn - Memory usage: {memory_usage:.2f} MB")
            print(f"scikit-learn - Accuracy: {accuracy:.2f}%")
            
            results.append({
                'framework': 'scikit-learn',
                'device': 'cpu',  # sklearn only uses CPU
                'train_time': train_time,
                'infer_time': infer_time,
                'memory_mb': memory_usage,
                'accuracy': accuracy,
                'epochs': args.epochs,
                'batch_size': args.batch_size,
            })
            
            # Clean up
            del model, predictions
            gc.collect()
            
        except Exception as e:
            print(f"Error testing scikit-learn: {e}")
    
    return results

def save_results(results):
    df = pd.DataFrame(results)

    # Create bar plots
    plt.figure(figsize=(12, 8))
    
    # Training time
    plt.subplot(2, 2, 1)
    frameworks = df['framework']
    train_times = df['train_time']
    plt.bar(frameworks, train_times)
    plt.title('Training Time (seconds)')
    plt.ylabel('Time (s)')
    plt.xticks(rotation=45)
    
    # Inference time
    plt.subplot(2, 2, 2)
    infer_times = df['infer_time']
    plt.bar(frameworks, infer_times)
    plt.title('Inference Time (seconds)')
    plt.ylabel('Time (s)')
    plt.xticks(rotation=45)
    
    # Memory usage
    plt.subplot(2, 2, 3)
    memory_usage = df['memory_mb']
    plt.bar(frameworks, memory_usage)
    plt.title('Memory Usage (MB)')
    plt.ylabel('Memory (MB)')
    plt.xticks(rotation=45)
    
    # Accuracy
    plt.subplot(2, 2, 4)
    accuracy = df['accuracy']
    plt.bar(frameworks, accuracy)
    plt.title('Accuracy (%)')
    plt.ylabel('Accuracy (%)')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, 'benchmark_results.png'))
    print(f"Plots saved to {os.path.join(args.output, 'benchmark_results.png')}")

if __name__ == "__main__":
    print(f"Starting benchmark with settings:")
    print(f"Dataset: {args.dataset}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Hidden size: {args.hidden_size}")
    print(f"Num layers: {args.num_layers}")
    print(f"Device: {args.device}")
    
    results = run_benchmark()
    
    if results:
        save_results(results)
        
        # Print summary table
        print("\n===== Benchmark Summary =====")
        df = pd.DataFrame(results)
        print(df[['framework', 'device', 'train_time', 'infer_time', 'accuracy']].to_string(index=False))
    else:
        print("No benchmark results to save")