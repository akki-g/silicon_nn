#include <metal_stdlib>
using namespace metal;

// Apple Silicon specific optimizations:
// 1. Adjusted tile sizes for Apple M-series GPUs (32 is often optimal)
// 2. Using simdgroup operations where available
// 3. More aggressive unrolling for the matrix multiplication inner loops
// 4. Added vectorized data loads and stores
// 5. Improved memory access patterns for better cache utilization

// Optimized tile size for Apple Silicon GPUs
#define TILE_SIZE 32

// Vectorized matrix multiplication kernel
// Computes C = A * Bᵀ + bias, where:
// - A is (M x K) - input matrix (batchSize x inputDim)
// - B is (N x K) - weight matrix (outputDim x inputDim) transposed
// - bias is (N) - bias vector (outputDim)
// - C is (M x N) - output matrix (batchSize x outputDim)
kernel void matrixMultiplyOptimized(
    device const float* A       [[ buffer(0) ]],  // Input matrix (M x K)
    device const float* B       [[ buffer(1) ]],  // Weight matrix (N x K)
    device const float* bias    [[ buffer(2) ]],  // Bias vector (N)
    device float* C             [[ buffer(3) ]],  // Output matrix (M x N)
    constant uint& M            [[ buffer(4) ]],  // Number of rows in A
    constant uint& K            [[ buffer(5) ]],  // Common dimension
    constant uint& N            [[ buffer(6) ]],  // Number of rows in B
    uint2 gid                   [[ thread_position_in_grid ]],
    uint2 tid                   [[ thread_position_in_threadgroup ]],
    uint2 groupSize             [[ threads_per_threadgroup ]],
    threadgroup float* sharedA  [[ threadgroup(0) ]],
    threadgroup float* sharedB  [[ threadgroup(1) ]])
{
    // Check boundaries
    if (gid.x >= M || gid.y >= N) {
        return;
    }
    
    // Compute threadgroup shared memory indices
    const uint localRow = tid.x;
    const uint localCol = tid.y;
    const uint numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    // Accumulate result for this thread
    float sum = 0.0f;
    
    // Loop over tiles
    for (uint t = 0; t < numTiles; t++) {
        // Load data into shared memory collaboratively
        uint globalK = t * TILE_SIZE + localCol;
        if (globalK < K) {
            // Load A tile (collaboratively)
            uint sharedIdxA = localRow * TILE_SIZE + localCol;
            sharedA[sharedIdxA] = (gid.x < M && globalK < K) ? A[gid.x * K + globalK] : 0.0f;
            
            // Load B tile (collaboratively)
            uint sharedIdxB = localRow * TILE_SIZE + localCol;
            sharedB[sharedIdxB] = (gid.y < N && globalK < K) ? B[gid.y * K + globalK] : 0.0f;
        }
        
        // Ensure all threads have loaded their data
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute partial dot product for this tile
        #pragma unroll 8
        for (uint k = 0; k < TILE_SIZE; k++) {
            if (t * TILE_SIZE + k < K) {
                sum += sharedA[localRow * TILE_SIZE + k] * sharedB[localCol * TILE_SIZE + k];
            }
        }
        
        // Ensure all computations are done before loading next tile
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Add bias and write final result
    if (gid.x < M && gid.y < N) {
        C[gid.x * N + gid.y] = sum + bias[gid.y];
    }
}

// Optimized dot product using simdgroup reduction for Apple Silicon
kernel void dotProductSIMD(
    device const float* A         [[ buffer(0) ]],
    device const float* B         [[ buffer(1) ]],
    device float* result          [[ buffer(2) ]],
    constant uint& count          [[ buffer(3) ]],
    uint gid                      [[ thread_position_in_grid ]],
    uint lid                      [[ thread_position_in_threadgroup ]],
    uint groupId                  [[ threadgroup_position_in_grid ]],
    uint simdLane                 [[ thread_index_in_simdgroup ]],
    uint simdGroup                [[ simdgroup_index_in_threadgroup ]],
    threadgroup float* localSums  [[ threadgroup(0) ]])
{
    constexpr uint simdSize = 32; // Size of SIMD group on Apple Silicon
    
    // Local accumulator
    float localSum = 0.0f;
    
    // Each thread processes multiple elements
    uint threadCount = min(simdSize * 32, count);
    uint elementsPerThread = (count + threadCount - 1) / threadCount;
    uint startIdx = gid * elementsPerThread;
    uint endIdx = min(startIdx + elementsPerThread, count);
    
    // Vector loads for better memory access pattern
    for (uint i = startIdx; i < endIdx; i += 4) {
        float4 aVec, bVec;
        if (i + 3 < endIdx) {
            aVec = float4(A[i], A[i+1], A[i+2], A[i+3]);
            bVec = float4(B[i], B[i+1], B[i+2], B[i+3]);
            localSum += dot(aVec, bVec);
        } else {
            for (uint j = i; j < endIdx; j++) {
                localSum += A[j] * B[j];
            }
        }
    }
    
    // Perform SIMD reduction within simdgroup (leverages hardware capabilities)
    float simdSum = simd_sum(localSum);
    
    // First thread in each SIMD group stores its result
    if (simdLane == 0) {
        localSums[simdGroup] = simdSum;
    }
    
    // Ensure all SIMD groups have written their result
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Final reduction (first SIMD group adds up all the partial sums)
    if (simdGroup == 0) {
        // Count number of active SIMD groups
        uint numSimdGroups = (threadCount + simdSize - 1) / simdSize;
        
        // First thread of first SIMD group initializes result
        if (simdLane == 0) {
            atomic_store_explicit((device atomic_float*)result, 0.0f, memory_order_relaxed);
        }
        
        // Each thread in first SIMD group adds one value
        if (simdLane < numSimdGroups) {
            float partialSum = localSums[simdLane];
            atomic_fetch_add_explicit((device atomic_float*)result, partialSum, memory_order_relaxed);
        }
    }
}

// Vectorized activation function application
kernel void applyActivationVectorized(
    device float* data           [[ buffer(0) ]],
    constant uint& count         [[ buffer(1) ]],
    constant uint& activationType [[ buffer(2) ]],
    uint gid                     [[ thread_position_in_grid ]])
{
    // Each thread processes 4 elements when possible for better vectorization
    uint idx = gid * 4;
    if (idx >= count) return;
    
    // Determine if we should process 4 elements or fewer
    bool processFull = (idx + 3 < count);
    
    if (processFull) {
        // Load 4 elements at once
        float4 values = float4(data[idx], data[idx+1], data[idx+2], data[idx+3]);
        
        // Apply activation function
        if (activationType == 0) {          // Sigmoid
            values = 1.0f / (1.0f + exp(-values));
        } else if (activationType == 1) {   // ReLU
            values = max(values, 0.0f);
        } else if (activationType == 2) {   // Tanh
            values = tanh(values);
        } else if (activationType == 4) {   // Leaky ReLU
            values = select(0.01f * values, values, values > 0.0f);
        } else if (activationType == 5) {   // ELU
            float4 zeros = float4(0.0f);
            values = select(float4(1.0f) * (exp(values) - 1.0f), values, values > zeros);
        }
        
        // Store results back
        data[idx] = values.x;
        data[idx+1] = values.y;
        data[idx+2] = values.z;
        data[idx+3] = values.w;
    } else {
        // Process remaining elements individually
        for (uint i = idx; i < count; i++) {
            float x = data[i];
            if (activationType == 0) {          // Sigmoid
                data[i] = 1.0f / (1.0f + exp(-x));
            } else if (activationType == 1) {   // ReLU
                data[i] = max(x, 0.0f);
            } else if (activationType == 2) {   // Tanh
                data[i] = tanh(x);
            } else if (activationType == 4) {   // Leaky ReLU
                data[i] = x > 0.0f ? x : 0.01f * x;
            } else if (activationType == 5) {   // ELU
                data[i] = x > 0.0f ? x : 1.0f * (exp(x) - 1.0f);
            }
        }
    }
}

// Specialized kernel for batch normalization
kernel void batchNorm(
    device const float* input     [[ buffer(0) ]],
    device const float* gamma     [[ buffer(1) ]],  // Scale parameter
    device const float* beta      [[ buffer(2) ]],  // Shift parameter
    device const float* mean      [[ buffer(3) ]],
    device const float* variance  [[ buffer(4) ]],
    device float* output          [[ buffer(5) ]],
    constant uint& batchSize      [[ buffer(6) ]],
    constant uint& featureSize    [[ buffer(7) ]],
    constant float& epsilon       [[ buffer(8) ]],
    uint2 gid                     [[ thread_position_in_grid ]])
{
    uint sampleIdx = gid.x;
    uint featureIdx = gid.y;
    
    if (sampleIdx >= batchSize || featureIdx >= featureSize) return;
    
    // Get input value
    float x = input[sampleIdx * featureSize + featureIdx];
    
    // Normalize
    float normalized = (x - mean[featureIdx]) / sqrt(variance[featureIdx] + epsilon);
    
    // Scale and shift
    float result = gamma[featureIdx] * normalized + beta[featureIdx];
    
    // Store result
    output[sampleIdx * featureSize + featureIdx] = result;
}

// Optimized kernel for softmax activation
kernel void softmax(
    device const float* input     [[ buffer(0) ]],
    device float* output          [[ buffer(1) ]],
    constant uint& batchSize      [[ buffer(2) ]],
    constant uint& featureSize    [[ buffer(3) ]],
    uint gid                      [[ thread_position_in_grid ]],
    threadgroup float* localMax   [[ threadgroup(0) ]],
    threadgroup float* localSum   [[ threadgroup(1) ]])
{
    uint sampleIdx = gid;
    if (sampleIdx >= batchSize) return;
    
    // Find maximum for numerical stability
    float maxVal = -INFINITY;
    for (uint i = 0; i < featureSize; i++) {
        float val = input[sampleIdx * featureSize + i];
        maxVal = max(maxVal, val);
    }
    localMax[0] = maxVal;
    
    // Compute exp and sum
    float sum = 0.0f;
    for (uint i = 0; i < featureSize; i++) {
        float expVal = exp(input[sampleIdx * featureSize + i] - maxVal);
        output[sampleIdx * featureSize + i] = expVal;
        sum += expVal;
    }
    localSum[0] = sum;
    
    // Normalize by sum
    for (uint i = 0; i < featureSize; i++) {
        output[sampleIdx * featureSize + i] /= sum;
    }
}