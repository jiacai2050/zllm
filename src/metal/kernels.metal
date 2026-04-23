#include <metal_stdlib>
using namespace metal;

kernel void rms_norm(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    device const float* weight [[buffer(2)]],
    constant float& eps [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    // Simplified RMSNorm for demonstration
    // In Task 5, we just want to verify dispatch works.
    dst[i] = src[i] * weight[i] + eps;
}
