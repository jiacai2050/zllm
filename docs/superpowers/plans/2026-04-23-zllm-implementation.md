# zLLM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a high-performance, standalone CLI inference engine for GGUF models on macOS using Zig 0.16.0 and Metal.

**Architecture:** A Zig-based application that parses GGUF models, manages memory statically, and dispatches compute work to Metal via a minimal C++ bridge using `metal-cpp`.

**Tech Stack:** Zig 0.16.0, Metal (MSL), C++ (metal-cpp bridge), GGUF format.

---

### Task 1: Project Scaffolding & Build System

**Files:**
- Create: `build.zig`
- Create: `src/main.zig`
- Create: `build.zig.zon`

- [ ] **Step 1: Create `build.zig.zon`**

```zig
.{
    .name = .zllm,
    .version = "0.1.0",
    .fingerprint = 0x58269e8893417721,
    .dependencies = .{},
    .paths = .{
        "build.zig",
        "build.zig.zon",
        "src",
    },
}
```

- [ ] **Step 2: Create a minimal `src/main.zig` with Juicy Main**

```zig
const std = @import("std");

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    try std.Io.File.stdout().writeStreamingAll(io, "zLLM starting...\n");
}
```

- [ ] **Step 3: Create `build.zig` with Metal/Foundation linkage**

```zig
const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const exe = b.addExecutable(.{
        .name = "zllm",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    exe.linkLibC();
    exe.linkFramework("Foundation");
    exe.linkFramework("Metal");
    exe.linkFramework("QuartzCore");

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    const run_step = b.step("run", "Run zLLM");
    run_step.dependOn(&run_cmd.step);
}
```

- [ ] **Step 4: Verify build and run**

Run: `zig build run`
Expected: Output `zLLM starting...`

- [ ] **Step 5: Commit**

```bash
git add build.zig build.zig.zon src/main.zig
git commit -m "feat: initial project scaffolding with Zig 0.16 juicy main"
```

---

### Task 2: Metal C++ Bridge & Device Initialization

**Files:**
- Create: `src/metal/bridge.h`
- Create: `src/metal/bridge.cpp`
- Modify: `build.zig`
- Create: `src/metal.zig`

- [ ] **Step 1: Create `src/metal/bridge.h` (C Interface)**

```c
#ifndef ZLLM_METAL_BRIDGE_H
#define ZLLM_METAL_BRIDGE_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ZLLM_Device ZLLM_Device;

ZLLM_Device* zllm_metal_init();
const char* zllm_metal_get_device_name(ZLLM_Device* device);
void zllm_metal_deinit(ZLLM_Device* device);

#ifdef __cplusplus
}
#endif

#endif
```

- [ ] **Step 2: Create `src/metal/bridge.cpp`**
Note: For now, we use standard Metal headers. `metal-cpp` can be added later if needed for complex objects.

```cpp
#include "bridge.h"
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

struct ZLLM_Device {
    id<MTLDevice> device;
};

ZLLM_Device* zllm_metal_init() {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) return nullptr;
    ZLLM_Device* d = (ZLLM_Device*)malloc(sizeof(ZLLM_Device));
    d->device = device;
    return d;
}

const char* zllm_metal_get_device_name(ZLLM_Device* device) {
    return [[device->device name] UTF8String];
}

void zllm_metal_deinit(ZLLM_Device* device) {
    free(device);
}
```

- [ ] **Step 3: Update `build.zig` to compile C++ bridge**

```zig
// Inside build.zig, add:
exe.addCSourceFile(.{
    .file = b.path("src/metal/bridge.cpp"),
    .flags = &.{ "-std=c++17", "-fobjc-arc" },
});
exe.addIncludePath(b.path("src/metal"));
exe.linkLibCpp();
```

- [ ] **Step 4: Create `src/metal.zig` wrapper**

```zig
const std = @import("std");

pub const Device = opaque {
    extern fn zllm_metal_init() ?*Device;
    extern fn zllm_metal_get_device_name(device: *Device) [*:0]const u8;
    extern fn zllm_metal_deinit(device: *Device) void;

    pub fn init() !*Device {
        return zllm_metal_init() orelse error.MetalInitFailed;
    }

    pub fn getName(self: *Device) []const u8 {
        return std.mem.span(zllm_metal_get_device_name(self));
    }

    pub fn deinit(self: *Device) void {
        zllm_metal_deinit(self);
    }
};
```

- [ ] **Step 5: Update `src/main.zig` to show device name**

```zig
const std = @import("std");
const metal = @import("metal.zig");

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const dev = try metal.Device.init();
    defer dev.deinit();

    var writer = std.Io.File.stdout().writer(&.{});
    try writer.interface.print("Metal Device: {s}\n", .{dev.getName()});
}
```

- [ ] **Step 6: Verify and commit**

Run: `zig build run`
Expected: Output includes "Metal Device: Apple M1" (or similar).
```bash
git add src/metal/bridge.h src/metal/bridge.cpp src/metal.zig src/main.zig build.zig
git commit -m "feat: add metal device initialization bridge"
```

---

### Task 3: GGUF Header Parsing

**Files:**
- Create: `src/gguf.zig`
- Create: `tests/gguf_test.zig`

- [ ] **Step 1: Define GGUF structures in `src/gguf.zig`**

```zig
const std = @import("std");

pub const Magic = [4]u8;
pub const GGUF_MAGIC: Magic = "GGUF".*;

pub const Header = struct {
    magic: Magic,
    version: u32,
    tensor_count: u64,
    metadata_kv_count: u64,
};

pub fn parseHeader(reader: anytype) !Header {
    var magic: Magic = undefined;
    try reader.readNoEof(&magic);
    if (!std.mem.eql(u8, &magic, &GGUF_MAGIC)) return error.InvalidMagic;

    return Header{
        .magic = magic,
        .version = try reader.readInt(u32, .little),
        .tensor_count = try reader.readInt(u64, .little),
        .metadata_kv_count = try reader.readInt(u64, .little),
    };
}
```

- [ ] **Step 2: Create unit test in `tests/gguf_test.zig`**

```zig
const std = @import("std");
const gguf = @import("../src/gguf.zig");

test "parse minimal gguf header" {
    var buf: [24]u8 = undefined;
    std.mem.copyForwards(u8, buf[0..4], "GGUF");
    std.mem.writeInt(u32, buf[4..8], 3, .little); // version
    std.mem.writeInt(u64, buf[8..16], 10, .little); // tensor_count
    std.mem.writeInt(u64, buf[16..24], 5, .little); // kv_count

    var fbs = std.io.fixedBufferStream(&buf);
    const header = try gguf.parseHeader(fbs.reader());

    try std.testing.expectEqual(header.version, 3);
    try std.testing.expectEqual(header.tensor_count, 10);
}
```

- [ ] **Step 3: Add test step to `build.zig`**

```zig
const unit_tests = b.addTest(.{
    .root_source_file = b.path("tests/gguf_test.zig"),
    .target = target,
    .optimize = optimize,
});
const run_unit_tests = b.addRunArtifact(unit_tests);
const test_step = b.step("test", "Run unit tests");
test_step.dependOn(&run_unit_tests.step);
```

- [ ] **Step 4: Verify and commit**

Run: `zig build test`
Expected: All tests passed.
```bash
git add src/gguf.zig tests/gguf_test.zig build.zig
git commit -m "feat: add GGUF header parsing and basic tests"
```

---

### Task 4: Tensor Metadata & Memory Mapping

**Files:**
- Modify: `src/gguf.zig`
- Modify: `src/main.zig`

- [ ] **Step 1: Implement String and Metadata Value parsing in `src/gguf.zig`**

```zig
pub const ValueType = enum(u32) {
    uint8 = 0, iint8 = 1, uint16 = 2, iint16 = 3, uint32 = 4, iint32 = 5,
    float32 = 6, bool = 7, string = 8, array = 9, uint64 = 10, iint64 = 11, float64 = 12,
};

pub const String = struct {
    len: u64,
    data: []const u8,

    pub fn parse(reader: anytype, allocator: std.mem.Allocator) !String {
        const len = try reader.readInt(u64, .little);
        const data = try allocator.alloc(u8, len);
        try reader.readNoEof(data);
        return String{ .len = len, .data = data };
    }
};
```

- [ ] **Step 2: Implement Tensor Info parsing in `src/gguf.zig`**

```zig
pub const TensorInfo = struct {
    name: String,
    n_dims: u32,
    dimensions: []u64,
    type: u32,
    offset: u64,
};

pub fn parseTensorInfo(reader: anytype, allocator: std.mem.Allocator) !TensorInfo {
    const name = try String.parse(reader, allocator);
    const n_dims = try reader.readInt(u32, .little);
    const dims = try allocator.alloc(u64, n_dims);
    for (dims) |*d| d.* = try reader.readInt(u64, .little);
    const dtype = try reader.readInt(u32, .little);
    const offset = try reader.readInt(u64, .little);
    return TensorInfo{ .name = name, .n_dims = n_dims, .dimensions = dims, .type = dtype, .offset = offset };
}
```

- [ ] **Step 3: Implement `mmap` loading in `src/main.zig`**

```zig
// Add to main.zig:
const file = try std.fs.cwd().openFile(model_path, .{});
defer file.close();
const stat = try file.stat();
const ptr = try std.posix.mmap(null, stat.size, std.posix.PROT.READ, .{ .TYPE = .SHARED }, file.handle, 0);
defer std.posix.munmap(ptr);
```

- [ ] **Step 4: Verify loading a small GGUF file**

Run: `zig build run -- model.gguf`
Expected: Successfully parse header and print tensor count.

---

### Task 5: Metal Compute Kernel Dispatch

**Files:**
- Create: `src/metal/kernels.metal`
- Modify: `src/metal/bridge.cpp`
- Modify: `src/metal/bridge.h`
- Modify: `src/metal.zig`

- [ ] **Step 1: Write a simple RMSNorm kernel in `src/metal/kernels.metal`**

```cpp
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
    dst[i] = src[i] * weight[i]; 
}
```

- [ ] **Step 2: Update bridge to load Shader Library in `src/metal/bridge.cpp`**

```cpp
struct ZLLM_Device {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> library;
};

ZLLM_Device* zllm_metal_init(const char* shader_source) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> queue = [device newCommandQueue];
    
    NSError* error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:shader_source] 
                                                  options:nil 
                                                    error:&error];
    ZLLM_Device* d = (ZLLM_Device*)malloc(sizeof(ZLLM_Device));
    d->device = device;
    d->queue = queue;
    d->library = library;
    return d;
}
```

- [ ] **Step 3: Add dispatch function to `src/metal/bridge.cpp`**

```cpp
void zllm_metal_dispatch(ZLLM_Device* d, const char* kernel_name, void** buffers, int n_buffers) {
    id<MTLFunction> func = [d->library newFunctionWithName:[NSString stringWithUTF8String:kernel_name]];
    id<MTLComputePipelineState> state = [d->device newComputePipelineStateWithFunction:func error:nil];
    id<MTLCommandBuffer> cmdBuf = [d->queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
    [encoder setComputePipelineState:state];
    for(int i=0; i<n_buffers; i++) {
        [encoder setBuffer:(id<MTLBuffer>)buffers[i] offset:0 atIndex:i];
    }
    // Dispatch threads...
    [encoder endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];
}
```

---

### Task 6: Inference Loop & Engine

**Files:**
- Create: `src/engine.zig`

- [ ] **Step 1: Define Engine structure in `src/engine.zig`**

```zig
const std = @import("std");
const metal = @import("metal.zig");

pub const Engine = struct {
    device: *metal.Device,
    tensors: std.StringHashMap(metal.Buffer),
    
    pub fn forward(self: *Engine, tokens: []const u32) ![]const f32 {
        // 1. Embedding
        // 2. Transformer Blocks (Attention + MLP)
        // 3. Output Logits
        return &.{};
    }
};
```

- [ ] **Step 2: Implement simple forward pass logic**

---

### Task 7: Quantization Support (Q4_K)

- [ ] **Step 1: Implement Q4_K dequantization in `src/metal/kernels.metal`**

```cpp
struct block_q4_K {
    half d;
    half dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};

kernel void matmul_q4_K(device const block_q4_K* x, ...) {
    // Implement Q4_K dot product logic
}
```

- [ ] **Step 2: Final CLI polish in `src/main.zig`**
- [ ] **Step 3: Verify with Llama-3-8B-Q4_K.gguf**
