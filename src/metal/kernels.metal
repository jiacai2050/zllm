#include <metal_stdlib>
using namespace metal;

// 1. RMSNorm
kernel void rms_norm(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    device const float* weight [[buffer(2)]],
    constant float& eps [[buffer(3)]],
    constant uint32_t& dim [[buffer(4)]],
    uint tg_idx [[thread_position_in_grid]]
) {
    float sum_sq = 0.0f;
    for (uint i = 0; i < dim; i++) {
        float val = src[i];
        sum_sq += val * val;
    }
    float rms = rsqrt(sum_sq / (float)dim + eps);
    for (uint i = 0; i < dim; i++) {
        dst[i] = src[i] * rms * weight[i];
    }
}

// 2. Embedding
struct block_q5_0 {
    half d;
    uint32_t qh;
    uint8_t qs[16];
};

kernel void embed_q5_0(
    device const uint32_t* tokens [[buffer(0)]],
    device float* dst [[buffer(1)]],
    device const uint8_t* embeddings [[buffer(2)]],
    constant uint32_t& dim [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    uint32_t token = tokens[0];
    uint32_t global_idx = token * dim + i;
    uint32_t block_idx = global_idx / 32;
    uint32_t k = global_idx % 32;
    device const uint8_t* b = embeddings + block_idx * 22;
    float d = (float)(*(device const half*)b);
    uint32_t qh = b[2] | (b[3] << 8) | (b[4] << 16) | (b[5] << 24);
    device const uint8_t* qs = b + 6;
    uint8_t q = qs[k % 16];
    uint8_t bits = (k < 16) ? (q & 0x0F) : (q >> 4);
    if (qh & (1U << k)) bits |= 0x10;
    dst[i] = d * ((float)bits - 16.0f);
}

kernel void embed_f32(
    device const uint32_t* tokens [[buffer(0)]],
    device float* dst [[buffer(1)]],
    device const float* embeddings [[buffer(2)]],
    constant uint32_t& dim [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    uint32_t token = tokens[0];
    dst[i] = embeddings[token * dim + i];
}

// 3. MatMul

kernel void matmul_q8_0(
    device const uint8_t* x [[buffer(0)]],
    device const float* y [[buffer(1)]],
    device float* dst [[buffer(2)]],
    constant uint32_t& ncols [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    float sum = 0.0f;
    uint32_t nb = ncols / 32;
    device const uint8_t* row = x + i * nb * 34;
    for (uint32_t j = 0; j < nb; j++) {
        device const uint8_t* b = row + j * 34;
        const float d = (float)(*(device const half*)b);
        device const int8_t* qs = (device const int8_t*)(b + 2);
        for (uint32_t k = 0; k < 32; k++) {
            sum += y[j * 32 + k] * d * (float)qs[k];
        }
    }
    dst[i] = sum;
}

kernel void matmul_q5_0(
    device const uint8_t* x [[buffer(0)]],
    device const float* y [[buffer(1)]],
    device float* dst [[buffer(2)]],
    constant uint32_t& ncols [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    float sum = 0.0f;
    uint32_t nb = ncols / 32;
    device const uint8_t* row = x + i * nb * 22;
    for (uint32_t j = 0; j < nb; j++) {
        device const uint8_t* b = row + j * 22;
        const float d = (float)(*(device const half*)b);
        uint32_t qh = b[2] | (b[3] << 8) | (b[4] << 16) | (b[5] << 24);
        device const uint8_t* qs = b + 6;
        for (uint32_t k = 0; k < 32; k++) {
            uint8_t q = qs[k % 16];
            uint8_t bits = (k < 16) ? (q & 0x0F) : (q >> 4);
            if (qh & (1U << k)) bits |= 0x10;
            sum += y[j * 32 + k] * d * ((float)bits - 16.0f);
        }
    }
    dst[i] = sum;
}

kernel void matmul_q4_0(
    device const uint8_t* x [[buffer(0)]],
    device const float* y [[buffer(1)]],
    device float* dst [[buffer(2)]],
    constant uint32_t& ncols [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    float sum = 0.0f;
    uint32_t nb = ncols / 32;
    device const uint8_t* row = x + i * nb * 18;
    for (uint32_t j = 0; j < nb; j++) {
        device const uint8_t* b = row + j * 18;
        const float d = (float)(*(device const half*)b);
        device const uint8_t* qs = b + 2;
        for (uint32_t k = 0; k < 16; k++) {
            uint8_t q = qs[k];
            sum += y[j * 32 + k] * d * ((float)(q & 0x0F) - 8.0f);
            sum += y[j * 32 + k + 16] * d * ((float)(q >> 4) - 8.0f);
        }
    }
    dst[i] = sum;
}

kernel void matmul_q4_K(
    device const uint8_t* x [[buffer(0)]],
    device const float* y [[buffer(1)]],
    device float* dst [[buffer(2)]],
    constant uint32_t& ncols [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    float sum = 0.0f;
    uint32_t nb = ncols / 256;
    device const uint8_t* row = x + i * nb * 144;
    for (uint32_t j = 0; j < nb; j++) {
        device const uint8_t* b = row + j * 144;
        float d = (float)(*(device const half*)b);
        float dmin = (float)(*(device const half*)(b + 2));
        device const uint8_t* qs = b + 16;
        for (uint32_t k = 0; k < 256; k++) {
            uint8_t q;
            if (k < 128) q = qs[k] & 0x0F;
            else q = qs[k - 128] >> 4;
            sum += y[j * 256 + k] * (d * q - dmin);
        }
    }
    dst[i] = sum;
}

kernel void matmul_q6_K(
    device const uint8_t* x [[buffer(0)]],
    device const float* y [[buffer(1)]],
    device float* dst [[buffer(2)]],
    constant uint32_t& ncols [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    float sum = 0.0f;
    uint32_t nb = ncols / 256;
    device const uint8_t* row = x + i * nb * 210;
    for (uint32_t j = 0; j < nb; j++) {
        device const uint8_t* b = row + j * 210;
        device const uint8_t* ql = b;
        float d = (float)(*(device const half*)(b + 208));
        for (uint32_t k = 0; k < 256; k++) {
            int8_t q = (k < 128) ? (ql[k] & 0xF) : (ql[k - 128] >> 4);
            sum += y[j * 256 + k] * d * (float)(q - 32);
        }
    }
    dst[i] = sum;
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

// 4. RoPE
kernel void rope(
    device float* src_dst [[buffer(0)]],
    constant uint32_t& pos [[buffer(1)]],
    constant uint32_t& head_dim [[buffer(2)]],
    constant float& freq_base [[buffer(3)]],
    uint head_idx [[thread_position_in_grid]]
) {
    uint32_t half_dim = head_dim / 2;
    for (uint32_t j = 0; j < half_dim; j++) {
        float theta = pow(freq_base, -((float)(2 * j) / (float)head_dim));
        float m_theta = (float)pos * theta;
        float cos_t = cos(m_theta);
        float sin_t = sin(m_theta);
        uint32_t idx0 = head_idx * head_dim + j;
        uint32_t idx1 = head_idx * head_dim + j + half_dim;
        float x0 = src_dst[idx0];
        float x1 = src_dst[idx1];
        src_dst[idx0] = x0 * cos_t - x1 * sin_t;
        src_dst[idx1] = x0 * sin_t + x1 * cos_t;
    }
}

// 5. Attention & KV Cache
kernel void update_kv_cache(
    device const float* k [[buffer(0)]],
    device const float* v [[buffer(1)]],
    device float* k_cache [[buffer(2)]],
    device float* v_cache [[buffer(3)]],
    constant uint32_t& pos [[buffer(4)]],
    constant uint32_t& n_layer [[buffer(5)]], // actually ilayer
    constant uint32_t& n_head_kv [[buffer(6)]],
    constant uint32_t& head_dim [[buffer(7)]],
    constant uint32_t& context_len [[buffer(8)]],
    uint i [[thread_position_in_grid]]
) {
    uint32_t ilayer = n_layer;
    uint32_t n_embd_kv = n_head_kv * head_dim;
    uint32_t cache_layer_offset = ilayer * n_embd_kv * context_len;
    k_cache[cache_layer_offset + pos * n_embd_kv + i] = k[i];
    v_cache[cache_layer_offset + pos * n_embd_kv + i] = v[i];
}

kernel void attention(
    device const float* q [[buffer(0)]],
    device const float* k_cache [[buffer(1)]],
    device const float* v_cache [[buffer(2)]],
    device float* dst [[buffer(3)]],
    constant uint32_t& pos [[buffer(4)]],
    constant uint32_t& ilayer [[buffer(5)]],
    constant uint32_t& n_head [[buffer(6)]],
    constant uint32_t& n_head_kv [[buffer(7)]],
    constant uint32_t& head_dim [[buffer(8)]],
    constant uint32_t& context_len [[buffer(9)]],
    uint ihead [[thread_position_in_grid]]
) {
    uint32_t n_embd_kv = n_head_kv * head_dim;
    uint32_t cache_layer_offset = ilayer * n_embd_kv * context_len;
    uint32_t kv_head_idx = ihead / (n_head / n_head_kv);
    
    device const float* head_q = q + ihead * head_dim;
    
    // 1. Q * K^T
    float scores[512]; // Limited to context_len 512
    float max_score = -1e20f;
    for (uint p = 0; p <= pos; p++) {
        float sum = 0.0f;
        device const float* head_k = k_cache + cache_layer_offset + p * n_embd_kv + kv_head_idx * head_dim;
        for (uint d = 0; d < head_dim; d++) {
            sum += head_q[d] * head_k[d];
        }
        sum /= sqrt((float)head_dim);
        scores[p] = sum;
        if (sum > max_score) max_score = sum;
    }
    
    // 2. Softmax
    float sum_exp = 0.0f;
    for (uint p = 0; p <= pos; p++) {
        scores[p] = exp(scores[p] - max_score);
        sum_exp += scores[p];
    }
    
    // 3. Score * V
    for (uint d = 0; d < head_dim; d++) {
        float val = 0.0f;
        for (uint p = 0; p <= pos; p++) {
            device const float* head_v = v_cache + cache_layer_offset + p * n_embd_kv + kv_head_idx * head_dim;
            val += (scores[p] / sum_exp) * head_v[d];
        }
        dst[ihead * head_dim + d] = val;
    }
}

// 6. SwiGLU
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

// 7. Add
kernel void add(
    device const float* x [[buffer(0)]],
    device float* dst [[buffer(1)]],
    uint i [[thread_position_in_grid]]
) {
    dst[i] += x[i];
}

// 8. Add Bias
kernel void add_bias(
    device const float* bias [[buffer(0)]],
    device float* dst [[buffer(1)]],
    uint i [[thread_position_in_grid]]
) {
    dst[i] += bias[i];
}

// 9. Copy
kernel void copy(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    uint i [[thread_position_in_grid]]
) {
    dst[i] = src[i];
}
