const std = @import("std");
const metal = @import("metal.zig");
const gguf = @import("gguf.zig");

pub const GGMLType = enum(u32) {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
};

fn getTensorSize(dtype: u32, dims: []u64) !u64 {
    var elements: u64 = 1;
    for (dims) |d| elements *= d;

    const ggml_type = std.enums.fromInt(GGMLType, dtype) orelse .F32;
    return switch (ggml_type) {
        .F32 => elements * 4,
        .F16 => elements * 2,
        .Q4_K => {
            if (elements % 256 != 0) return error.InvalidQuantizedDimensions;
            return (elements / 256) * 144;
        },
        else => elements * 4,
    };
}

pub const Engine = struct {
    device: *metal.Device,
    model: gguf.Model,
    tensors: std.StringArrayHashMapUnmanaged(*metal.Buffer),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, device: *metal.Device, model: gguf.Model, mmap_buffer: []const u8) !*Engine {
        var tensors: std.StringArrayHashMapUnmanaged(*metal.Buffer) = .empty;
        errdefer {
            var it = tensors.iterator();
            while (it.next()) |entry| {
                entry.value_ptr.*.release();
            }
            tensors.deinit(allocator);
        }

        for (model.tensors) |t| {
            const offset = model.data_offset + t.offset;
            const size = try getTensorSize(t.type, t.dimensions);
            
            if (offset + size > mmap_buffer.len) return error.TensorOutOfBounds;
            
            const buf = try device.createBuffer(mmap_buffer[offset..offset+size], size);
            try tensors.put(allocator, t.name.data, buf);
        }

        const self = try allocator.create(Engine);
        self.* = Engine{
            .device = device,
            .model = model,
            .tensors = tensors,
            .allocator = allocator,
        };
        return self;
    }

    pub fn deinit(self: *Engine) void {
        var it = self.tensors.iterator();
        while (it.next()) |entry| {
            entry.value_ptr.*.release();
        }
        self.tensors.deinit(self.allocator);
        self.allocator.destroy(self);
    }

    pub fn forward(self: *Engine, tokens: []const u32) ![]const f32 {
        _ = self;
        _ = tokens;
        // Placeholder for now
        return &.{};
    }
};
