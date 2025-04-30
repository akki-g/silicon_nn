#import "MetalDevice.h"
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <iostream>
#include <Eigen/Dense>
#include "../common/ActivationFunctions.h"
#include <cmath>

using Eigen::MatrixXd;
using Eigen::VectorXd;

MetalDevice::MetalDevice() {
    metalDevice = MTLCreateSystemDefaultDevice();
    if (metalDevice) {
        commandQueue = [metalDevice newCommandQueue];
        
        // Load the Metal library containing our optimized kernels
        NSError* error = nil;
        NSString* kernelPath = [[NSBundle mainBundle] pathForResource:@"OptimizedKernels" ofType:@"metallib"];
        
        if (kernelPath) {
            // Use newer API to avoid deprecation warning
            NSURL* url = [NSURL fileURLWithPath:kernelPath];
            defaultLibrary = [metalDevice newLibraryWithURL:url error:&error];
        } else {
            // Fall back to compiled default library if metallib file is not found
            defaultLibrary = [metalDevice newDefaultLibrary];
        }
        
        if (!defaultLibrary) {
            NSLog(@"Failed to load Metal library: %@", error);
        }
        
        std::cout << "MetalDevice created successfully with " 
                  << (defaultLibrary ? "optimized kernels" : "default kernels") << std::endl;
    } else {
        std::cerr << "Failed to create Metal device" << std::endl;
        exit(1);
    }
}

MetalDevice::~MetalDevice() {
    if (!isCleaned) {
        cleanup();
    }
    std::cout << "MetalDevice destroyed" << std::endl;
}

double MetalDevice::dot(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) {
        std::cerr << "Vector dimensions don't match for dot product" << std::endl;
        exit(1);
    }
    
    int n = a.size();
    
    // For small vectors, just use CPU
    if (n < 1000) {
        double result = 0.0;
        for (int i = 0; i < n; i++) {
            result += a[i] * b[i];
        }
        return result;
    }
    
    // For larger vectors, use GPU with optimized SIMD kernel
    
    // Convert to float arrays
    std::vector<float> a_float(n);
    std::vector<float> b_float(n);
    for (int i = 0; i < n; i++) {
        a_float[i] = static_cast<float>(a[i]);
        b_float[i] = static_cast<float>(b[i]);
    }
    
    // Create Metal buffers
    id<MTLBuffer> bufferA = [metalDevice newBufferWithBytes:a_float.data() 
                                                    length:n * sizeof(float) 
                                                    options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> bufferB = [metalDevice newBufferWithBytes:b_float.data() 
                                                    length:n * sizeof(float) 
                                                    options:MTLResourceStorageModeShared];
    
    // Result buffer (single float)
    id<MTLBuffer> bufferResult = [metalDevice newBufferWithLength:sizeof(float) 
                                                        options:MTLResourceStorageModeShared];
    
    // Set initial result to 0
    float initialResult = 0.0f;
    memcpy([bufferResult contents], &initialResult, sizeof(float));
    
    // Get kernel function - use optimized SIMD version for Apple Silicon
    id<MTLFunction> dotFunction = [defaultLibrary newFunctionWithName:@"dotProductSIMD"];
    if (!dotFunction) {
        NSLog(@"Failed to find optimized dot product kernel function, falling back");
        dotFunction = [defaultLibrary newFunctionWithName:@"dotProductOptimized"];
        if (!dotFunction) {
            NSLog(@"No dot product kernel found, performing computation on CPU");
            double result = 0.0;
            for (int i = 0; i < n; i++) {
                result += a[i] * b[i];
            }
            return result;
        }
    }
    
    // Create compute pipeline
    NSError* error = nil;
    id<MTLComputePipelineState> pipeline = [metalDevice newComputePipelineStateWithFunction:dotFunction 
                                                                                    error:&error];
    
    if (!pipeline) {
        NSLog(@"Failed to create pipeline state: %@", error);
        return 0.0;
    }
    
    // Create command buffer and encoder
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    
    // Set pipeline and buffers
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:bufferA offset:0 atIndex:0];
    [encoder setBuffer:bufferB offset:0 atIndex:1];
    [encoder setBuffer:bufferResult offset:0 atIndex:2];
    
    // Set vector size as parameter
    uint32_t count = static_cast<uint32_t>(n);
    [encoder setBytes:&count length:sizeof(uint32_t) atIndex:3];
    
    // Calculate threadgroup size and grid size
    NSUInteger maxThreads = pipeline.maxTotalThreadsPerThreadgroup;
    NSUInteger threadgroupSize = maxThreads < 32 ? maxThreads : 32; // SIMD size is 32 on Apple Silicon
    NSUInteger numThreads = (n + 127) / 128; // Each thread processes multiple elements
    
    // Allocate threadgroup memory for partial sums
    uint32_t numSimdGroups = (threadgroupSize + 31) / 32;
    [encoder setThreadgroupMemoryLength:numSimdGroups * sizeof(float) atIndex:0];
    
    // Dispatch threads
    [encoder dispatchThreads:MTLSizeMake(numThreads, 1, 1) 
        threadsPerThreadgroup:MTLSizeMake(threadgroupSize, 1, 1)];
    
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    // Get result
    float result;
    memcpy(&result, [bufferResult contents], sizeof(float));
    
    return static_cast<double>(result);
}

std::vector<double> MetalDevice::matmul(const std::vector<double>& A, const std::vector<double>& B,
                                        int A_rows, int A_cols, int B_cols) {
    // For simplicity, we'll use the Eigen version
    MatrixXd a = Eigen::Map<const MatrixXd>(A.data(), A_rows, A_cols);
    MatrixXd b = Eigen::Map<const MatrixXd>(B.data(), A_cols, B_cols);
    
    MatrixXd c = a * b;
    
    std::vector<double> result(A_rows * B_cols);
    Eigen::Map<MatrixXd>(result.data(), A_rows, B_cols) = c;
    
    return result;
}

void MetalDevice::applyActivation(std::vector<double>& data, double (*activation)(double)) {
    int n = data.size();
    
    // For small arrays, just use CPU
    if (n < 1000) {
        for (int i = 0; i < n; i++) {
            data[i] = activation(data[i]);
        }
        return;
    }
    
    // For larger arrays, use GPU with vectorized activation
    
    // Convert to float array
    std::vector<float> data_float(n);
    for (int i = 0; i < n; i++) {
        data_float[i] = static_cast<float>(data[i]);
    }
    
    // Create Metal buffer
    id<MTLBuffer> buffer = [metalDevice newBufferWithBytes:data_float.data() 
                                                   length:n * sizeof(float) 
                                                  options:MTLResourceStorageModeShared];
    
    // Determine activation type through behavior pattern
    uint32_t activationType = 0;  // Default to sigmoid
    
    // Test values to identify the function
    double testValue1 = 1.0;   // Positive value
    double testValue2 = -1.0;  // Negative value
    
    double result1 = activation(testValue1);
    double result2 = activation(testValue2);
    
    // Identify activation function by its behavior
    if (std::abs(result1 - 1.0) < 1e-6 && std::abs(result2) < 1e-6) {
        // ReLU behavior: f(1) = 1, f(-1) = 0
        activationType = 1;
    } else if (std::abs(result1 - std::tanh(1.0)) < 1e-6 && std::abs(result2 - std::tanh(-1.0)) < 1e-6) {
        // Tanh behavior: f(1) ≈ 0.76, f(-1) ≈ -0.76
        activationType = 2;
    } else if (result2 < 0.0 && result2 > -0.1) {
        // Leaky ReLU behavior: f(-1) is small negative (with default alpha of 0.01)
        activationType = 4;
    } else if (result2 < 0.0 && result2 > -0.7) {
        // ELU behavior: f(-1) = -0.63 with default alpha of 1.0
        activationType = 5;
    } else {
        // Default to sigmoid
        activationType = 0;
    }
    
    // Get vectorized activation kernel
    id<MTLFunction> activationFunction = [defaultLibrary newFunctionWithName:@"applyActivationVectorized"];
    if (!activationFunction) {
        NSLog(@"Failed to find vectorized activation kernel, falling back to basic implementation");
        activationFunction = [defaultLibrary newFunctionWithName:@"applyActivationOptimized"];
        if (!activationFunction) {
            NSLog(@"No activation kernel found, performing computation on CPU");
            for (int i = 0; i < n; i++) {
                data[i] = activation(data[i]);
            }
            return;
        }
    }
    
    // Create compute pipeline
    NSError* error = nil;
    id<MTLComputePipelineState> pipeline = [metalDevice newComputePipelineStateWithFunction:activationFunction 
                                                                                     error:&error];
    
    if (!pipeline) {
        NSLog(@"Failed to create pipeline state: %@", error);
        return;
    }
    
    // Create command buffer and encoder
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    
    // Set pipeline and buffers
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buffer offset:0 atIndex:0];
    
    // Set parameters
    uint32_t count = static_cast<uint32_t>(n);
    [encoder setBytes:&count length:sizeof(uint32_t) atIndex:1];
    [encoder setBytes:&activationType length:sizeof(uint32_t) atIndex:2];
    
    // Calculate threadgroup size and grid size
    NSUInteger maxThreads = pipeline.maxTotalThreadsPerThreadgroup;
    NSUInteger threadgroupSize = maxThreads < 256 ? maxThreads : 256;
    
    // Each thread processes 4 elements for vectorization
    NSUInteger numThreads = (n + 3) / 4;
    
    // Dispatch threads
    [encoder dispatchThreads:MTLSizeMake(numThreads, 1, 1) 
         threadsPerThreadgroup:MTLSizeMake(threadgroupSize, 1, 1)];
    
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    // Copy results back
    float* result_ptr = (float*)[buffer contents];
    for (int i = 0; i < n; i++) {
        data[i] = static_cast<double>(result_ptr[i]);
    }
}

MatrixXd MetalDevice::matmulGPU(const MatrixXd &A, const MatrixXd &W, const VectorXd &b) {
    int m = A.rows();    // batch size
    int n = A.cols();    // input dimension
    int k = W.rows();    // output dimension
    
    // Check if optimized Metal kernels are available
    if (!defaultLibrary) {
        NSLog(@"Metal library not available, falling back to CPU");
        MatrixXd Z = A * W.transpose();
        Z.rowwise() += b.transpose();
        return Z;
    }
    
    // Try to use the optimized matrix multiply kernel for Apple Silicon
    id<MTLFunction> matMulFunction = [defaultLibrary newFunctionWithName:@"matrixMultiplyOptimized"];
    if (!matMulFunction) {
        NSLog(@"Optimized matrix multiply kernel not found, trying fallback");
        matMulFunction = [defaultLibrary newFunctionWithName:@"matMulTiled"];
        if (!matMulFunction) {
            NSLog(@"No matrix multiplication kernel found, falling back to CPU");
            MatrixXd Z = A * W.transpose();
            Z.rowwise() += b.transpose();
            return Z;
        }
    }
    
    // Convert to float arrays
    std::vector<float> A_data(m * n);
    std::vector<float> W_data(k * n);
    std::vector<float> b_data(k);
    std::vector<float> Z_data(m * k, 0.0f);
    
    // Copy data from Eigen to float arrays
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            A_data[i * n + j] = static_cast<float>(A(i, j));
        }
    }
    
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            W_data[i * n + j] = static_cast<float>(W(i, j));
        }
        b_data[i] = static_cast<float>(b(i));
    }
    
    // Create Metal buffers
    id<MTLBuffer> bufferA = [metalDevice newBufferWithBytes:A_data.data() 
                                                   length:A_data.size() * sizeof(float) 
                                                  options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> bufferW = [metalDevice newBufferWithBytes:W_data.data() 
                                                   length:W_data.size() * sizeof(float) 
                                                  options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> bufferB = [metalDevice newBufferWithBytes:b_data.data() 
                                                    length:b_data.size() * sizeof(float) 
                                                    options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> bufferZ = [metalDevice newBufferWithBytes:Z_data.data()
                                                    length:Z_data.size() * sizeof(float) 
                                                    options:MTLResourceStorageModeShared];
    
    // Create compute pipeline
    NSError* error = nil;
    id<MTLComputePipelineState> pipeline = [metalDevice newComputePipelineStateWithFunction:matMulFunction 
                                                                                     error:&error];
    
    if (!pipeline) {
        NSLog(@"Failed to create pipeline state: %@", error);
        MatrixXd Z = A * W.transpose();
        Z.rowwise() += b.transpose();
        return Z;
    }
    
    // Create command buffer and encoder
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    
    // Set pipeline and buffers
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:bufferA offset:0 atIndex:0];
    [encoder setBuffer:bufferW offset:0 atIndex:1];
    [encoder setBuffer:bufferB offset:0 atIndex:2];
    [encoder setBuffer:bufferZ offset:0 atIndex:3];
    
    // Set dimensions as parameters
    uint32_t M = static_cast<uint32_t>(m);
    uint32_t K = static_cast<uint32_t>(n); // K is input dimension
    uint32_t N = static_cast<uint32_t>(k); // N is output dimension
    [encoder setBytes:&M length:sizeof(uint32_t) atIndex:4];
    [encoder setBytes:&K length:sizeof(uint32_t) atIndex:5];
    [encoder setBytes:&N length:sizeof(uint32_t) atIndex:6];
    
    // Calculate threadgroup size and shared memory
    const int TILE_SIZE = 32; // Optimized for Apple Silicon
    MTLSize threadgroupSize = MTLSizeMake(TILE_SIZE, TILE_SIZE, 1);
    
    // Allocate shared memory for tiles
    [encoder setThreadgroupMemoryLength:TILE_SIZE * TILE_SIZE * sizeof(float) atIndex:0]; // For A
    [encoder setThreadgroupMemoryLength:TILE_SIZE * TILE_SIZE * sizeof(float) atIndex:1]; // For B
    
    // Dispatch threads - one thread per output element
    MTLSize gridSize = MTLSizeMake(m, k, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    // Copy result back to Eigen matrix
    float* Z_ptr = (float*)[bufferZ contents];
    MatrixXd Z(m, k);
    
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            // Bias is already added in the kernel
            Z(i, j) = static_cast<double>(Z_ptr[i * k + j]);
        }
    }
    
    return Z;
}

// Implementation of batch normalization on GPU
MatrixXd MetalDevice::batchNormGPU(const MatrixXd& input, const VectorXd& gamma, 
                                   const VectorXd& beta, const VectorXd& mean,
                                   const VectorXd& var, double epsilon) {
    int batchSize = input.rows();
    int featureSize = input.cols();
    
    // Check if kernel is available
    if (!defaultLibrary) {
        NSLog(@"Metal library not available, falling back to CPU for batch norm");
        return batchNormCPU(input, gamma, beta, mean, var, epsilon);
    }
    
    id<MTLFunction> batchNormFunction = [defaultLibrary newFunctionWithName:@"batchNorm"];
    if (!batchNormFunction) {
        NSLog(@"Batch norm kernel not found, falling back to CPU");
        return batchNormCPU(input, gamma, beta, mean, var, epsilon);
    }
    
    // Convert data to float arrays
    std::vector<float> inputData(batchSize * featureSize);
    std::vector<float> gammaData(featureSize);
    std::vector<float> betaData(featureSize);
    std::vector<float> meanData(featureSize);
    std::vector<float> varData(featureSize);
    std::vector<float> outputData(batchSize * featureSize);
    
    // Copy input data
    for (int i = 0; i < batchSize; i++) {
        for (int j = 0; j < featureSize; j++) {
            inputData[i * featureSize + j] = static_cast<float>(input(i, j));
        }
    }
    
    // Copy parameters
    for (int j = 0; j < featureSize; j++) {
        gammaData[j] = static_cast<float>(gamma(j));
        betaData[j] = static_cast<float>(beta(j));
        meanData[j] = static_cast<float>(mean(j));
        varData[j] = static_cast<float>(var(j));
    }
    
    // Create Metal buffers
    id<MTLBuffer> inputBuffer = [metalDevice newBufferWithBytes:inputData.data()
                                                        length:inputData.size() * sizeof(float)
                                                       options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> gammaBuffer = [metalDevice newBufferWithBytes:gammaData.data()
                                                        length:gammaData.size() * sizeof(float)
                                                       options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> betaBuffer = [metalDevice newBufferWithBytes:betaData.data()
                                                       length:betaData.size() * sizeof(float)
                                                      options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> meanBuffer = [metalDevice newBufferWithBytes:meanData.data()
                                                       length:meanData.size() * sizeof(float)
                                                      options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> varBuffer = [metalDevice newBufferWithBytes:varData.data()
                                                      length:varData.size() * sizeof(float)
                                                     options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> outputBuffer = [metalDevice newBufferWithLength:outputData.size() * sizeof(float)
                                                         options:MTLResourceStorageModeShared];
    
    // Create compute pipeline
    NSError* error = nil;
    id<MTLComputePipelineState> pipeline = [metalDevice newComputePipelineStateWithFunction:batchNormFunction
                                                                                     error:&error];
    
    if (!pipeline) {
        NSLog(@"Failed to create pipeline for batch norm: %@", error);
        return batchNormCPU(input, gamma, beta, mean, var, epsilon);
    }
    
    // Create command buffer and encoder
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    
    // Set pipeline and buffers
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:inputBuffer offset:0 atIndex:0];
    [encoder setBuffer:gammaBuffer offset:0 atIndex:1];
    [encoder setBuffer:betaBuffer offset:0 atIndex:2];
    [encoder setBuffer:meanBuffer offset:0 atIndex:3];
    [encoder setBuffer:varBuffer offset:0 atIndex:4];
    [encoder setBuffer:outputBuffer offset:0 atIndex:5];
    
    // Set parameters
    uint32_t batchSizeUint = static_cast<uint32_t>(batchSize);
    uint32_t featureSizeUint = static_cast<uint32_t>(featureSize);
    float epsilonFloat = static_cast<float>(epsilon);
    
    [encoder setBytes:&batchSizeUint length:sizeof(uint32_t) atIndex:6];
    [encoder setBytes:&featureSizeUint length:sizeof(uint32_t) atIndex:7];
    [encoder setBytes:&epsilonFloat length:sizeof(float) atIndex:8];
    
    // Dispatch threads - one thread per output element
    MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);
    MTLSize gridSize = MTLSizeMake(batchSize, featureSize, 1);
    
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    // Copy result back to Eigen matrix
    float* result_ptr = (float*)[outputBuffer contents];
    MatrixXd output(batchSize, featureSize);
    
    for (int i = 0; i < batchSize; i++) {
        for (int j = 0; j < featureSize; j++) {
            output(i, j) = static_cast<double>(result_ptr[i * featureSize + j]);
        }
    }
    
    return output;
}

// CPU fallback for batch normalization
MatrixXd MetalDevice::batchNormCPU(const MatrixXd& input, const VectorXd& gamma, 
                                  const VectorXd& beta, const VectorXd& mean,
                                  const VectorXd& var, double epsilon) {
    int batchSize = input.rows();
    int featureSize = input.cols();
    
    MatrixXd output(batchSize, featureSize);
    
    for (int i = 0; i < batchSize; i++) {
        for (int j = 0; j < featureSize; j++) {
            double normalized = (input(i, j) - mean(j)) / std::sqrt(var(j) + epsilon);
            output(i, j) = gamma(j) * normalized + beta(j);
        }
    }
    
    return output;
}

// Implementation of softmax on GPU
MatrixXd MetalDevice::softmaxGPU(const MatrixXd& input) {
    int batchSize = input.rows();
    int featureSize = input.cols();
    
    // Check if kernel is available
    if (!defaultLibrary) {
        NSLog(@"Metal library not available, falling back to CPU for softmax");
        return softmaxCPU(input);
    }
    
    id<MTLFunction> softmaxFunction = [defaultLibrary newFunctionWithName:@"softmax"];
    if (!softmaxFunction) {
        NSLog(@"Softmax kernel not found, falling back to CPU");
        return softmaxCPU(input);
    }
    
    // Convert data to float arrays
    std::vector<float> inputData(batchSize * featureSize);
    std::vector<float> outputData(batchSize * featureSize);
    
    // Copy input data
    for (int i = 0; i < batchSize; i++) {
        for (int j = 0; j < featureSize; j++) {
            inputData[i * featureSize + j] = static_cast<float>(input(i, j));
        }
    }
    
    // Create Metal buffers
    id<MTLBuffer> inputBuffer = [metalDevice newBufferWithBytes:inputData.data()
                                                        length:inputData.size() * sizeof(float)
                                                       options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> outputBuffer = [metalDevice newBufferWithLength:outputData.size() * sizeof(float)
                                                         options:MTLResourceStorageModeShared];
    
    // Create compute pipeline
    NSError* error = nil;
    id<MTLComputePipelineState> pipeline = [metalDevice newComputePipelineStateWithFunction:softmaxFunction
                                                                                     error:&error];
    
    if (!pipeline) {
        NSLog(@"Failed to create pipeline for softmax: %@", error);
        return softmaxCPU(input);
    }
    
    // Create command buffer and encoder
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    
    // Set pipeline and buffers
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:inputBuffer offset:0 atIndex:0];
    [encoder setBuffer:outputBuffer offset:0 atIndex:1];
    
    // Set parameters
    uint32_t batchSizeUint = static_cast<uint32_t>(batchSize);
    uint32_t featureSizeUint = static_cast<uint32_t>(featureSize);
    
    [encoder setBytes:&batchSizeUint length:sizeof(uint32_t) atIndex:2];
    [encoder setBytes:&featureSizeUint length:sizeof(uint32_t) atIndex:3];
    
    // Allocate threadgroup memory
    [encoder setThreadgroupMemoryLength:sizeof(float) atIndex:0]; // For max value
    [encoder setThreadgroupMemoryLength:sizeof(float) atIndex:1]; // For sum
    
    // Dispatch threads - one thread per sample
    NSUInteger threadgroupSize = pipeline.maxTotalThreadsPerThreadgroup < 32 ? 
                                pipeline.maxTotalThreadsPerThreadgroup : 32;
    
    [encoder dispatchThreads:MTLSizeMake(batchSize, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threadgroupSize, 1, 1)];
    
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    // Copy result back to Eigen matrix
    float* result_ptr = (float*)[outputBuffer contents];
    MatrixXd output(batchSize, featureSize);
    
    for (int i = 0; i < batchSize; i++) {
        for (int j = 0; j < featureSize; j++) {
            output(i, j) = static_cast<double>(result_ptr[i * featureSize + j]);
        }
    }
    
    return output;
}

// CPU fallback for softmax
MatrixXd MetalDevice::softmaxCPU(const MatrixXd& input) {
    int batchSize = input.rows();
    int featureSize = input.cols();
    
    MatrixXd output(batchSize, featureSize);
    
    for (int i = 0; i < batchSize; i++) {
        // Find max for numerical stability
        double maxVal = input.row(i).maxCoeff();
        
        // Compute exp(x_i - max) and sum
        double sum = 0.0;
        for (int j = 0; j < featureSize; j++) {
            output(i, j) = std::exp(input(i, j) - maxVal);
            sum += output(i, j);
        }
        
        // Normalize
        output.row(i) /= sum;
    }
    
    return output;
}

bool MetalDevice::metalIsAvailable() {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    return (dev != nil);
}

void MetalDevice::cleanup() {
    if (isCleaned) return;
    
    std::cout << "MetalDevice: Cleaning up resources" << std::endl;
    
    if (commandQueue) {
        commandQueue = nil;
    }
    
    if (defaultLibrary) {
        defaultLibrary = nil;
    }
    
    if (metalDevice) {
        metalDevice = nil;
    }
    
    isCleaned = true;
}