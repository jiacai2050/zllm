# zLLM Design Document

## 1. Introduction
zLLM is a lightweight, high-performance inference engine for Apple Silicon macOS, implemented in Zig 0.16.0. It aims to provide a standalone CLI tool for running GGUF-formatted LLMs using Metal for hardware acceleration.

## 2. Goals
- **Performance**: Leverage Apple Silicon's unified memory and Metal SIMD instructions.
- **Simplicity**: Zero external dependencies beyond macOS system Frameworks.
- **Modern Zig**: Adhere to Zig 0.16.0 standards (Juicy Main, std.Io) and TigerStyle guidelines.

## 3. Architecture

### 3.1. Overview
The system is divided into three main layers:
1.  **Zig Application Layer**: Handles CLI, GGUF parsing, memory management, and high-level inference logic.
2.  **C Bridge Layer**: A minimal wrapper around `metal-cpp` to expose Metal functionality to Zig.
3.  **Metal Hardware Layer**: Custom MSL (Metal Shading Language) kernels for optimized tensor operations.

### 3.2. Core Components
-   **Juicy Main Entry**: Uses `std.process.Init` to manage lifecycle and resources.
-   **GGUF Parser**: Pure Zig implementation to parse metadata and tensor info from `.gguf` files.
-   **Static Memory Allocator**: Pre-calculates and allocates all `MTLBuffer`s upfront (weights + activation buffers + KV cache).
-   **Metal Engine**: Controls the execution pipeline, submitting compute commands to the GPU.

## 4. Implementation Details

### 4.1. Memory Management
-   **Zero-Copy**: Use `mmap` for GGUF files and map memory directly to `MTLBuffer` with `MTLStorageModeShared`.
-   **Static Pipelines**: No runtime allocations during the inference loop.

### 4.2. Metal Kernels
-   **Custom MSL**: Implementation of core LLM operators (RMSNorm, RoPE, MatMul/GEMV).
-   **Quantization Support**: Native support for GGUF quantization formats (e.g., Q4_K, Q8_0) within shaders.

### 4.3. Shader Compilation (Hybrid Mode)
-   **Development**: JIT compilation from source strings for fast iteration.
-   **Production**: Pre-compiled `.metallib` embedded into the executable.

## 5. Development Plan
1.  Setup project structure and `build.zig` with Metal linkage.
2.  Implement GGUF parser in Zig.
3.  Implement C/C++ bridge for Metal.
4.  Write basic Metal kernels (RMSNorm, Simple MatMul).
5.  Assemble the inference loop and CLI.
6.  Optimize kernels for quantized formats.

## 6. Testing Strategy
-   Unit tests for GGUF parsing logic.
-   Kernel validation against reference CPU implementations.
-   End-to-end integration tests with small models (e.g., Llama-3-8B-Q4_K).
