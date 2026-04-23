# zLLM Implementation Guide: From Theory to Code

This document maps the concepts from `prerequisites.md` directly into the `zLLM` codebase. We will explore how Zig 0.16.0 and Apple's Metal API are used to build a high-performance, zero-dependency inference engine.

---

## 1. Codebase Architecture Overview
The engine is split into highly specialized, decoupled modules:

*   **`src/main.zig`**: The orchestrator. Handles file loading, memory mapping, and the core generation loop.
*   **`src/gguf.zig`**: The parser. Reads the GGUF file format to extract model weights and metadata (like layer counts).
*   **`src/tokenizer.zig`**: The translator. Converts user text into Token IDs and vice-versa.
*   **`src/engine.zig`**: The architect. Manages all GPU memory buffers and builds the Transformer Computation Graph.
*   **`src/metal/kernels.metal`**: The workers. The raw C++ programs running on thousands of GPU cores.
*   **`src/sampler.zig`**: The decision maker. Analyzes the final probabilities to pick the next word.

---

## 2. Zero-Copy Loading (`src/main.zig`)
On a traditional PC, loading a 4GB model involves reading it from the SSD into RAM, then copying it over the PCIe bus into the GPU's VRAM.

**The Apple Silicon Advantage (Unified Memory):**
Because the CPU and GPU share the exact same physical memory chips, we can skip the copying entirely.

In `main.zig`, we use `mmap` (Memory Map):
```zig
const mmap_buffer = try std.posix.mmap(
    null,
    model_file_size,
    .{ .READ = true },
    .{ .TYPE = .SHARED },
    file.handle,
    0,
);
```
This tells the OS: *"Treat this file on disk as if it were RAM"*.
When we pass this `mmap_buffer` to Metal, the GPU reads the weights directly from the OS page cache. **Zero copying is performed.**

---

## 3. The Engine & Static Memory (`src/engine.zig`)
Memory allocation (`malloc`) is slow. Doing it during the generation loop will ruin performance.
`zLLM` uses a **Static Allocation Strategy**.

During `Engine.init()`, we ask the GGUF metadata for the model's exact shape (e.g., 896 embedding length, 24 layers). We then ask the GPU to pre-allocate all the "scratchpad" buffers we will ever need.

### The KV Cache Allocation:
The short-term memory (KV Cache) needs to hold the Keys and Values for every layer, every head, for the entire context length (e.g., 512 tokens).
```zig
const k_cache = try device.createBuffer(null, layer_count * head_count_kv * head_dimension * context_length * @sizeOf(f32));
```
Once allocated, `engine.zig` never asks for memory again. The `forward` function simply routes data through these pre-existing pipes.

---

## 4. The Computation Graph (`src/engine.zig`)
If you look at the `computeLayer` function in `engine.zig`, it maps 1:1 with the Transformer diagram from the prerequisites.

Instead of writing `for` loops in Zig to multiply matrices, the engine **dispatches commands to the GPU**.
```zig
// 1. RMSNorm
self.device.dispatch("rms_norm", &.{ self.hidden_states, self.norm_states, ... }, 1);

// 2. Self-Attention (QKV Projections)
try self.dispatchMatMul(q_weight, self.norm_states, self.q, self.embedding_length);

// 3. Update the KV Cache
self.device.dispatch("update_kv_cache", ...);

// 4. MLP (SwiGLU)
self.device.dispatch("swiglu", &.{ self.mlp_gate, self.mlp_up, self.mlp_gate }, ...);
```

---

## 5. The GPU Workers (`src/metal/kernels.metal`)
While `engine.zig` is the architect, `kernels.metal` are the actual workers.
A Kernel is a small C++ function designed to be run by thousands of GPU threads simultaneously.

### Example: The MatMul Kernel (`matmul_f16`)
When `engine.zig` calls `dispatchMatMul`, Metal spins up hundreds of threads.
Notice the `uint i [[thread_position_in_grid]]`. This is the thread's unique ID.

If we are calculating a 896x896 matrix multiplication, thread `0` calculates row `0`, thread `1` calculates row `1`, entirely in parallel.

```cpp
kernel void matmul_f16(
    device const half* x [[buffer(0)]], // The Model Weights
    device const float* y [[buffer(1)]], // The Input Vector
    device float* dst [[buffer(2)]],    // The Output Vector
    constant uint32_t& ncols [[buffer(3)]],
    uint i [[thread_position_in_grid]]  // Thread ID!
) {
    float sum = 0.0f;
    device const half* row = x + i * ncols; 
    
    // Each thread does the dot-product for its specific row.
    for (uint32_t j = 0; j < ncols; j++) {
        sum += (float)row[j] * y[j];
    }
    dst[i] = sum;
}
```

### The Quantization Decompression
For a quantized model (e.g., `Q8_0`), the kernel does extra work on the fly. It reads the 8-bit integer, multiplies it by the `f16` scale stored in the block, and *then* does the math. This trades a tiny bit of GPU compute for a massive saving in memory bandwidth.

---

## 6. Top-P Sampling (`src/sampler.zig`)
Once the Engine finishes passing the data through all 24 layers, it spits out `logits`—a raw score for every possible word in the 151,936-word dictionary.

We could just pick the word with the highest score (Argmax). But that makes the AI sound like a boring, repetitive robot.

**Top-P (Nucleus) Sampling**:
1. We convert raw scores to percentages (Softmax).
2. We sort the words from highest probability to lowest.
3. We keep adding words to a "bucket" until their combined probability hits a threshold (e.g., `P = 0.9` or 90%).
4. We randomly roll a dice and pick a word *only* from that bucket.

This allows the model to occasionally pick a less common word (making it creative) while completely ignoring nonsensical words (keeping it coherent).
