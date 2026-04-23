const std = @import("std");
const gguf = @import("gguf.zig");

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    var args = init.minimal.args.iterate();
    _ = args.next();
    const model_path = args.next() orelse return error.MissingModelPath;

    const file = try std.Io.Dir.cwd().openFile(io, model_path, .{});
    defer file.close(io);

    const stat = try file.stat(io);
    const buffer = try std.posix.mmap(
        null,
        stat.size,
        .{ .READ = true },
        .{ .TYPE = .SHARED },
        file.handle,
        0,
    );
    defer std.posix.munmap(buffer);

    var reader: std.Io.Reader = .fixed(buffer);
    const header = try gguf.parseHeader(&reader);

    for (0..header.metadata_kv_count) |_| {
        const kv = try gguf.MetadataKV.parse(&reader, allocator);
        defer kv.deinit(allocator);

        if (std.mem.eql(u8, kv.key.data, "tokenizer.ggml.tokens")) {
            std.debug.print("First 10 tokens:\n", .{});
            for (kv.value.array.data[0..10], 0..) |v, i| {
                std.debug.print("  {d}: {s}\n", .{ i, v.string.data });
            }
        }
    }
}
