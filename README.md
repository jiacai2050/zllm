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

### 2. Download the Recommended Model
For the best experience (clean English output), download the **Qwen2.5-0.5B-Instruct FP16** model:

- **Download Link**: [qwen2.5-0.5b-instruct-fp16.gguf](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/blob/main/qwen2.5-0.5b-instruct-fp16.gguf)

### 3. Build and Run

#### Direct Prompt Mode
```bash
zig build -Doptimize=ReleaseFast
./zig-out/bin/zllm /path/to/qwen2.5-0.5b-instruct-fp16.gguf -p "What is the capital of France?"
```

#### Interactive Chat Mode
```bash
./zig-out/bin/zllm /path/to/qwen2.5-0.5b-instruct-fp16.gguf
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
