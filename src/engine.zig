const std = @import("std");
const metal = @import("metal.zig");
const gguf = @import("gguf.zig");

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
            
            var elements: u64 = 1;
            for (t.dimensions) |d| elements *= d;
            
            // TODO: Proper size calculation for quantized types
            const size = elements * 4; // assume f32 for now
            
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
