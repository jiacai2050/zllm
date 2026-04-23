# zLLM - LLM Inference Engine

## Project Overview
**zLLM** is a lightweight, high-performance Large Language Model (LLM) inference engine designed specifically for macOS Apple Silicon. It is built entirely with **Zig 0.16.0** and **Metal** API, featuring a Zero-Copy Architecture via memory mapping (`mmap`) to load GGUF models directly to unified memory, minimizing CPU-GPU data transfer overhead. It natively supports block-based quantization formats (like Q4_K and Q8_0) directly in GPU shaders and avoids any heavy external dependencies, relying solely on macOS system frameworks (Foundation, Metal, QuartzCore).

## Building and Running

### Prerequisites
- **macOS** with Apple Silicon (M1/M2/M3/M4)
- **Zig 0.16.0** (Required for features like "Juicy Main" via `std.process.Init` and unified `std.Io`)

### Commands
- **Build the project:**
  ```bash
  zig build -Doptimize=ReleaseFast
  ```
- **Run inference (Direct Prompt Mode):**
  ```bash
  ./zig-out/bin/zllm /path/to/model.gguf -p "Your prompt here"
  ```
  *Alternatively using `zig build run`:*
  ```bash
  zig build run -Doptimize=ReleaseFast -- /path/to/model.gguf -p "Your prompt here"
  ```
- **Run inference (Interactive Chat Mode):**
  ```bash
  ./zig-out/bin/zllm /path/to/model.gguf
  ```
- **Run tests:**
  ```bash
  zig build test
  ```

## Development Conventions

zLLM strictly adheres to the **TigerStyle** Zig coding guidelines, prioritizing safety, performance, and clarity:
- **Naming Constraints:** No abbreviations are allowed (e.g., use `embedding_length` instead of `n_embd`). Use `camelCase` for functions, `PascalCase` for types, and `snake_case` for variables.
- **Safety and Assertions:** Defensive programming is mandatory. Rely heavily on paired `std.debug.assert` at function entry points to validate assumptions, boundaries, and alignment.
- **Error Handling:** All errors must be explicitly handled. Discarding errors (e.g., `_ = err`) is prohibited.
- **Control Flow:** Keep functions as small and "flat" as possible. Extract deep nesting or hot paths (like Attention or MLP blocks) into explicit private functions to improve compilation caching and clarity.
- **Memory Strategy:** The engine uses a strict **Static Memory Allocation** strategy. All required intermediate buffers and KV caches are pre-allocated during `Engine.init`. Memory allocation (`malloc`/`allocator.alloc`) during the inference loop (`forward`) is forbidden.
- **Formatting:** Code must be formatted using `zig fmt` and strictly conform to a 100-column line limit.

## Key Directories and Files
- `src/main.zig`: The application orchestrator and unified generation loop entry point.
- `src/engine.zig`: The core engine handling static memory management and Transformer Computation Graph.
- `src/gguf.zig`: Pure Zig parser for GGUF metadata and tensor information.
- `src/tokenizer.zig`: BPE text-to-token translator handling specific byte-level mapping cleanup.
- `src/metal/`: Contains Objective-C++ bridge (`bridge.mm`) and MSL GPU compute kernels (`kernels.metal`).
- `docs/`: Comprehensive theoretical (`prerequisites.md`) and engineering (`implementation.md`) manuals.
