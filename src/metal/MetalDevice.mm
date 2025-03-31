#import "MetalDevice.h"
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <iostream>
#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;

MetalDevice::MetalDevice() {
    metalDevice = MTLCreateSystemDefaultDevice();
    if (metalDevice) {
        commandQueue = [metalDevice newCommandQueue];
        std::cout << "MetalDevice created successfully" << std::endl;
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
    
    // For larger vectors, use GPU
    
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
    
    // Get kernel function
    id<MTLLibrary> defaultLibrary = [metalDevice newDefaultLibrary];
    if (!defaultLibrary) {
        NSLog(@"Failed to load Metal library");
        return 0.0;
    }
    
    id<MTLFunction> dotFunction = [defaultLibrary newFunctionWithName:@"dotProductOptimized"];
    if (!dotFunction) {
        NSLog(@"Failed to find dot product kernel function");
        return 0.0;
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
    NSUInteger threadgroupSize = std::min(pipeline.maxTotalThreadsPerThreadgroup, 256u);
    NSUInteger gridSize = (n + threadgroupSize - 1) / threadgroupSize * threadgroupSize;
    
    // Allocate threadgroup memory for partial sums
    [encoder setThreadgroupMemoryLength:threadgroupSize * sizeof(float) atIndex:0];
    
    // Dispatch threads
    [encoder dispatchThreads:MTLSizeMake(gridSize, 1, 1) 
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
    
    // For larger arrays, use GPU
    
    // Convert to float array
    std::vector<float> data_float(n);
    for (int i = 0; i < n; i++) {
        data_float[i] = static_cast<float>(data[i]);
    }
    
    // Create Metal buffer
    id<MTLBuffer> buffer = [metalDevice newBufferWithBytes:data_float.data() 
                                                   length:n * sizeof(float) 
                                                  options:MTLResourceStorageModeShared];
    
    // Determine activation type
    uint32_t activationType = 0;  // Default to sigmoid
    
    if (activation == relu) {
        activationType = 1;
    } else if (activation == tanhActivation) {
        activationType = 2;
    }
    
    // Get kernel function
    id<MTLLibrary> defaultLibrary = [metalDevice newDefaultLibrary];
    if (!defaultLibrary) {
        NSLog(@"Failed to load Metal library");
        return;
    }
    
    id<MTLFunction> activationFunction = [defaultLibrary newFunctionWithName:@"applyActivationOptimized"];
    if (!activationFunction) {
        NSLog(@"Failed to find activation kernel function");
        return;
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
    NSUInteger threadgroupSize = std::min(pipeline.maxTotalThreadsPerThreadgroup, 256u);
    NSUInteger gridSize = (n + threadgroupSize - 1) / threadgroupSize;
    
    // Dispatch threads
    [encoder dispatchThreads:MTLSizeMake(n, 1, 1) 
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
    
    // Check if Metal shaders are available
    id<MTLLibrary> defaultLibrary = [metalDevice newDefaultLibrary];
    if (!defaultLibrary) {
        NSLog(@"Failed to load Metal library, falling back to CPU");
        MatrixXd Z = A * W.transpose();
        Z.rowwise() += b.transpose();
        return Z;
    }
    
    id<MTLFunction> matMulFunction = [defaultLibrary newFunctionWithName:@"matMulTiled"];
    if (!matMulFunction) {
        NSLog(@"Failed to find matrix multiplication kernel, falling back to CPU");
        MatrixXd Z = A * W.transpose();
        Z.rowwise() += b.transpose();
        return Z;
    }
    
    // Convert to float arrays
    std::vector<float> A_data(m * n);
    std::vector<float> W_data(k * n);
    std::vector<float> b_data(k);
    std::vector<float> Z_data(m * k);
    
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
    
    id<MTLBuffer> bufferZ = [metalDevice newBufferWithLength:Z_data.size() * sizeof(float) 
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
    [encoder setBuffer:bufferZ offset:0 atIndex:2];
    
    // Set dimensions as parameters
    uint32_t M = static_cast<uint32_t>(m);
    uint32_t N = static_cast<uint32_t>(n);
    uint32_t K = static_cast<uint32_t>(k);
    [encoder setBytes:&M length:sizeof(uint32_t) atIndex:3];
    [encoder setBytes:&N length:sizeof(uint32_t) atIndex:4];
    [encoder setBytes:&K length:sizeof(uint32_t) atIndex:5];
    
    // Calculate threadgroup size
    const int TILE_SIZE = 16;
    MTLSize threadgroupSize = MTLSizeMake(TILE_SIZE, TILE_SIZE, 1);
    MTLSize gridSize = MTLSizeMake(m, k, 1);
    
    // Dispatch threads
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    // Copy result back to Eigen matrix
    float* Z_ptr = (float*)[bufferZ contents];
    MatrixXd Z(m, k);
    
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            // Add bias term
            Z(i, j) = static_cast<double>(Z_ptr[i * k + j]) + b(j);
        }
    }
    
    return Z;
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
    
    if (metalDevice) {
        metalDevice = nil;
    }
    
    isCleaned = true;
}