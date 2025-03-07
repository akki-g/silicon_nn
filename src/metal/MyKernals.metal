#include <metal_stdlib>
using namespace metal;

kernal void vectorAdd(
    device const float* inA [[ buffer(0) ]],
    device const float* inB [[ buffer(1) ]],
    device float* outC [[ buffer(2) ]],
    uint id [[ thread_position_in_grid ]]
) {
    outC[id] = inA[id] + inB[id];
}