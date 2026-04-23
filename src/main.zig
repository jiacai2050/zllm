const std = @import("std");
const metal = @import("metal.zig");
const gguf = @import("gguf.zig");
const engine = @import("engine.zig");

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const allocator = init.gpa;

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
    const model = try gguf.parse(&reader, allocator);
    defer model.deinit(allocator);

    std.debug.print("GGUF version: {d}\n", .{model.header.version});
    std.debug.print("Tensor count: {d}\n", .{model.header.tensor_count});
    std.debug.print("Data offset: {d}\n", .{model.data_offset});

    const kernels_source = @embedFile("metal/kernels.metal");
    const dev = try metal.Device.init(kernels_source);
    defer dev.deinit();

    std.debug.print("Metal Device: {s}\n", .{dev.getName()});

    const e = try engine.Engine.init(allocator, dev, model, buffer);
    defer e.deinit();

    std.debug.print("Engine initialized with {d} tensors.\n", .{e.tensors.count()});
}
