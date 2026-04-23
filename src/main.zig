const std = @import("std");
const metal = @import("metal.zig");
const gguf = @import("gguf.zig");
const engine = @import("engine.zig");

fn readLine(reader: *std.Io.Reader, buffer: []u8) !?[]u8 {
    var i: usize = 0;
    while (i < buffer.len) {
        var b: [1]u8 = undefined;
        const n = try reader.readSliceShort(&b);
        if (n == 0) return if (i == 0) null else buffer[0..i];
        if (b[0] == '\n') return buffer[0..i];
        if (b[0] == '\r') continue;
        buffer[i] = b[0];
        i += 1;
    }
    return buffer[0..i];
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const allocator = init.gpa;

    var args = init.minimal.args.iterate();
    _ = args.next(); // skip program name
    const model_path = args.next() orelse {
        std.debug.print("Usage: zllm <model_path>\n", .{});
        return;
    };

    var out_buf: [4096]u8 = undefined;
    var stdout = std.Io.File.stdout().writer(io, &out_buf);
    
    try stdout.interface.print("--- zLLM Inference Engine ---\n", .{});
    try stdout.interface.print("Loading model: {s}\n", .{model_path});

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

    try stdout.interface.print("GGUF version: {d}\n", .{model.header.version});
    try stdout.interface.print("Tensor count: {d}\n", .{model.header.tensor_count});

    const kernels_source = @embedFile("metal/kernels.metal");
    const dev = try metal.Device.init(kernels_source);
    defer dev.deinit();

    try stdout.interface.print("Metal Device: {s}\n", .{dev.getName()});

    const e = try engine.Engine.init(allocator, dev, model, buffer);
    defer e.deinit();

    try stdout.interface.print("Engine initialized with {d} tensors.\n", .{e.tensors.count()});
    try stdout.interface.print("\nWelcome to zLLM! Enter your prompt below (type 'exit' to quit).\n", .{});

    var line_buf: [1024]u8 = undefined;
    var in_buf: [1024]u8 = undefined;
    var stdin = std.Io.File.stdin().reader(io, &in_buf);
    
    while (true) {
        try stdout.interface.print("> ", .{});
        // Flush stdout
        try stdout.flush();
        
        const line = try readLine(&stdin.interface, &line_buf) orelse break;
        const trimmed = std.mem.trim(u8, line, " \r\n");
        if (trimmed.len == 0) continue;
        if (std.mem.eql(u8, trimmed, "exit") or std.mem.eql(u8, trimmed, "quit")) break;

        try stdout.interface.print("zLLM: (Thinking...)\n", .{});
        try stdout.flush();
        
        // Placeholder for forward pass
        _ = try e.forward(&.{1}); 
        try stdout.interface.print("zLLM: Quantization support is active. Matmul Q4_K is ready in shaders.\n", .{});
        try stdout.flush();
    }
}
