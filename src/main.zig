const std = @import("std");
const metal = @import("metal.zig");
const gguf = @import("gguf.zig");

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    var args = init.minimal.args.iterate();
    _ = args.next(); // skip program name
    const model_path = args.next() orelse {
        std.debug.print("Usage: zllm <model_path>\n", .{});
        return;
    };

    std.debug.print("Loading model: {s}\n", .{model_path});

    const file = try std.Io.Dir.cwd().openFile(io, model_path, .{});
    defer file.close(io);

    const stat = try file.stat(io);
    const size = stat.size;
    std.debug.print("Model size: {d}, fd: {d}\n", .{size, file.handle});

    const buffer = try std.posix.mmap(
        null,
        size,
        .{ .READ = true },
        .{ .TYPE = .SHARED },
        file.handle,
        0,
    );
    defer std.posix.munmap(buffer);

    var reader: std.Io.Reader = .fixed(buffer);
    const header = try gguf.parseHeader(&reader);

    std.debug.print("GGUF version: {d}\n", .{header.version});
    std.debug.print("Tensor count: {d}\n", .{header.tensor_count});
    std.debug.print("Metadata KV count: {d}\n", .{header.metadata_kv_count});

    // Task 5: Initialize Metal with kernels
    const kernels_source = @embedFile("metal/kernels.metal");
    const dev = try metal.Device.init(kernels_source);
    defer dev.deinit();

    std.debug.print("Metal Device: {s}\n", .{dev.getName()});

    // Simple test for RMSNorm kernel
    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const weight = [_]f32{ 1.0, 1.0, 1.0, 1.0 };
    var eps: f32 = 1e-6;

    const buf_src = try dev.createBuffer(std.mem.sliceAsBytes(&data), @sizeOf(f32) * 4);
    defer buf_src.release();
    const buf_dst = try dev.createBuffer(null, @sizeOf(f32) * 4);
    defer buf_dst.release();
    const buf_weight = try dev.createBuffer(std.mem.sliceAsBytes(&weight), @sizeOf(f32) * 4);
    defer buf_weight.release();
    const buf_eps = try dev.createBuffer(std.mem.asBytes(&eps), @sizeOf(f32));
    defer buf_eps.release();

    const buffers = [_]?*metal.Buffer{ buf_src, buf_dst, buf_weight, buf_eps };
    dev.dispatch("rms_norm", &buffers, 4);

    std.debug.print("RMSNorm kernel dispatched successfully.\n", .{});
}
