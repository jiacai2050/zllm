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

kernel void embed(
    device const uint32_t* tokens [[buffer(0)]],
    device float* dst [[buffer(1)]],
    device const float* embeddings [[buffer(2)]],
    constant uint32_t& dim [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    uint32_t token_idx = i / dim;
    uint32_t dim_idx = i % dim;
    uint32_t token = tokens[token_idx];
    dst[i] = embeddings[token * dim + dim_idx];
}

struct block_q4_K {
    half d;
    half dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};

kernel void matmul_q4_K(
    device const block_q4_K* x [[buffer(0)]],
    device const float* y [[buffer(1)]],
    device float* dst [[buffer(2)]],
    constant uint32_t& ncols [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    float sum = 0.0f;
    uint32_t nb = ncols / 256;
    
    device const block_q4_K* b = x + i * nb;
    device const float* vy = y;

    for (uint32_t j = 0; j < nb; j++) {
        const float d = (float)b[j].d;
        const float dmin = (float)b[j].dmin;
        
        for (uint32_t k = 0; k < 256; k++) {
            uint8_t q = b[j].qs[k/2];
            q = (k%2 == 0) ? (q & 0x0F) : (q >> 4);
            // Simplified dequantization for demo
            sum += vy[k] * (d * q - dmin);
        }
        vy += 256;
    }
    dst[i] = sum;
}
