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

    const dev = try metal.Device.init();
    defer dev.deinit();

    std.debug.print("Metal Device: {s}\n", .{dev.getName()});
}
