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

kernel void rope(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant uint32_t& pos [[buffer(2)]],
    constant uint32_t& head_dim [[buffer(3)]],
    constant float& freq_base [[buffer(4)]],
    uint i [[thread_position_in_grid]]
) {
    uint32_t half_dim = head_dim / 2;
    uint32_t head_idx = i / half_dim;
    uint32_t j = i % half_dim;

    float theta = pow(freq_base, -((float)(2 * j) / (float)head_dim));
    float m_theta = (float)pos * theta;
    float cos_t = cos(m_theta);
    float sin_t = sin(m_theta);

    uint32_t idx0 = head_idx * head_dim + j;
    uint32_t idx1 = head_idx * head_dim + j + half_dim;

    float x0 = src[idx0];
    float x1 = src[idx1];

    dst[idx0] = x0 * cos_t - x1 * sin_t;
    dst[idx1] = x0 * sin_t + x1 * cos_t;
}

kernel void swiglu(
    device const float* x [[buffer(0)]],
    device const float* y [[buffer(1)]],
    device float* dst [[buffer(2)]],
    uint i [[thread_position_in_grid]]
) {
    float val = x[i];
    float silu = val / (1.0f + exp(-val));
    dst[i] = silu * y[i];
}

kernel void matmul_f32(
    device const float* x [[buffer(0)]],
    device const float* y [[buffer(1)]],
    device float* dst [[buffer(2)]],
    constant uint32_t& ncols [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    float sum = 0.0f;
    device const float* row = x + i * ncols;
    for (uint32_t j = 0; j < ncols; j++) {
        sum += row[j] * y[j];
    }
    dst[i] = sum;
}
