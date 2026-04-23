# zLLM

A lightweight, high-performance LLM inference engine for macOS Apple Silicon, built with **Zig 0.16.0** and **Metal**.

## Features

- **Native Metal Acceleration**: Custom Metal Shading Language (MSL) kernels for optimized tensor operations (RMSNorm, Embedding, MatMul).
- **Zig 0.16 "Juicy Main"**: Leveraging the latest Zig standard library features including `std.process.Init` and the unified `std.Io` interface.
- **Zero-Copy Architecture**: Uses `mmap` to map GGUF model files directly into Metal buffers, minimizing CPU-GPU data transfer overhead.
- **GGUF Support**: Pure Zig parser for the GGUF file format, supporting metadata and tensor information extraction.
- **Quantization**: Built-in support for GGUF quantization formats like **Q4_K** and **Q8_0** implemented directly in GPU shaders.
- **Zero Dependencies**: Built entirely with Zig and macOS system frameworks (Foundation, Metal, QuartzCore). No heavy external libraries required.

## Prerequisites

- **macOS** (with Apple Silicon M1/M2/M3/M4 recommended)
- **Zig 0.16.0**

## Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/jiacai2050/zllm.git
cd zllm
```

### 2. Build the project
```bash
zig build
```

### 3. Run with a model
To run the inference engine, you need a GGUF format model. You can also generate a dummy model for testing:

```bash
# Generate a dummy model
zig run scripts/create_dummy.zig

# Run zLLM
./zig-out/bin/zllm dummy.gguf
```

## Project Structure

- `src/main.zig`: Entry point using Zig 0.16 Juicy Main.
- `src/gguf.zig`: Pure Zig GGUF format parser.
- `src/metal.zig`: Zig wrapper for the Metal C++ bridge.
- `src/engine.zig`: Inference engine logic and memory management.
- `src/metal/`:
    - `bridge.mm`: Objective-C++ bridge to Metal API.
    - `kernels.metal`: High-performance GPU compute kernels.

## Implementation Details

zLLM follows a **Static Memory Allocation** strategy. All required buffers (weights, KV cache, activations) are calculated and allocated during model loading. This ensures zero runtime allocations during the inference loop, providing predictable and low-latency performance.

## License

MIT
