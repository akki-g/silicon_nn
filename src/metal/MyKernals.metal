#include <metal_stdlib>
using namespace metal;

// Tune tile size based on your GPU; 16 or 32 are common.
#define TILE_SIZE 16

// Optimized tiled matrix multiplication kernel.
// Computes C = A * Bᵀ, where A is (M x N) and B is (K x N), so C is (M x K).
// (Each row of B represents a neuron’s weights.)
kernel void matMulTiled(
    device const float* A       [[ buffer(0) ]],  // Input matrix A (M x N)
    device const float* B       [[ buffer(1) ]],  // Weight matrix B (K x N) stored row-major
    device float*       C       [[ buffer(2) ]],  // Output matrix C (M x K)
    constant uint&      M       [[ buffer(3) ]],
    constant uint&      N       [[ buffer(4) ]],
    constant uint&      K       [[ buffer(5) ]],
    uint2 gid                   [[ thread_position_in_grid ]],
    uint2 tid                   [[ thread_position_in_threadgroup ]],
    threadgroup float tileA[TILE_SIZE][TILE_SIZE],
    threadgroup float tileB[TILE_SIZE][TILE_SIZE])
{
    // Calculate the row and column of the element of C to compute.
    uint row = gid.x;
    uint col = gid.y;
    float sum = 0.0;
    
    // Loop over tiles.
    for (uint t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load a tile from A into shared memory.
        uint tiledCol = t * TILE_SIZE + tid.y;
        if (row < M && tiledCol < N)
            tileA[tid.x][tid.y] = A[row * N + tiledCol];
        else
            tileA[tid.x][tid.y] = 0.0;
        
        // Load a tile from B into shared memory.
        // Note: B is stored row-major with dimensions (K x N). We want Bᵀ so that
        // each tile from B is transposed in shared memory.
        uint tiledRow = t * TILE_SIZE + tid.x;
        if (tiledRow < N && col < K)
            tileB[tid.x][tid.y] = B[col * N + tiledRow]; // effectively Bᵀ[col, tiledRow]
        else
            tileB[tid.x][tid.y] = 0.0;
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Multiply the two tiles together.
        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += tileA[tid.x][k] * tileB[k][tid.y];
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write the computed value to C if within bounds.
    if (row < M && col < K)
        C[row * K + col] = sum;
}

// Optimized dot product kernel.
// Computes a dot product between vectors A and B (of length "count").
// Each thread computes a partial sum and then a reduction is performed.
kernel void dotProductOptimized(
    device const float* A       [[ buffer(0) ]],
    device const float* B       [[ buffer(1) ]],
    device float*       result  [[ buffer(2) ]],
    constant uint&      count   [[ buffer(3) ]],
    uint tid                    [[ thread_index_in_threadgroup ]],
    uint gid                    [[ thread_position_in_grid ]],
    threadgroup float* partialSums [[ threadgroup(0) ]])
{
    float sum = 0.0;
    // Each thread sums over its strided portion.
    for (uint i = gid; i < count; i += grid_size) {
        sum += A[i] * B[i];
    }
    partialSums[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduction within the threadgroup.
    uint t = threadgroup_size / 2;
    while(t > 0) {
        if (tid < t) {
            partialSums[tid] += partialSums[tid + t];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        t /= 2;
    }
    if (tid == 0) {
        // Use atomic addition if multiple threadgroups are used.
        atomic_fetch_add_explicit((device atomic_float*)result, partialSums[0], memory_order_relaxed);
    }
}

// Element-wise activation kernel.
// Applies an activation function to an array of "count" elements.
kernel void applyActivationOptimized(
    device float* data          [[ buffer(0) ]],
    constant uint& count        [[ buffer(1) ]],
    constant uint& activationType [[ buffer(2) ]],
    uint gid                    [[ thread_position_in_grid ]])
{
    if (gid < count) {
        float x = data[gid];
        if (activationType == 0) {          // Sigmoid
            data[gid] = 1.0 / (1.0 + exp(-x));
        } else if (activationType == 1) {   // ReLU
            data[gid] = (x > 0.0) ? x : 0.0;
        } else if (activationType == 2) {   // Tanh
            data[gid] = tanh(x);
        } else {
            // For SOFTMAX, activation is usually computed as part of the matmul.
            data[gid] = x;
        }
    }
}
