const std = @import("std");
const metal = @import("metal.zig");
const gguf = @import("gguf.zig");
const engine = @import("engine.zig");
const tokenizer = @import("tokenizer");
const sampler = @import("sampler.zig");

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

    var model_path: ?[]const u8 = null;
    var prompt: ?[]const u8 = null;

    var args = init.minimal.args.iterate();
    _ = args.next(); // skip program name
    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "-p") or std.mem.eql(u8, arg, "--prompt")) {
            prompt = args.next() orelse {
                std.debug.print("Error: -p/--prompt requires a value\n", .{});
                return;
            };
        } else if (model_path == null) {
            model_path = arg;
        } else {
            // Ignore other arguments for now
        }
    }

    const path = model_path orelse {
        std.debug.print("Usage: zllm <model_path> [-p <prompt>]\n", .{});
        return;
    };

    var out_buf: [4096]u8 = undefined;
    var stdout = std.Io.File.stdout().writer(io, &out_buf);

    try stdout.interface.print("--- zLLM Inference Engine ---\n", .{});
    try stdout.interface.print("Loading model: {s}\n", .{path});

    const file = try std.Io.Dir.cwd().openFile(io, path, .{});
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

    // Initialize Tokenizer
    const tokens_kv = model.findMetadata("tokenizer.ggml.tokens") orelse {
        std.debug.print("Error: model is missing 'tokenizer.ggml.tokens' metadata\n", .{});
        return error.MissingTokens;
    };
    if (tokens_kv.value != .array or tokens_kv.value.array.type != .string) {
        std.debug.print("Error: 'tokenizer.ggml.tokens' is not a string array\n", .{});
        return error.InvalidTokensType;
    }

    const tokens_array = tokens_kv.value.array.data;
    var token_strings = try allocator.alloc([]const u8, tokens_array.len);
    defer allocator.free(token_strings);
    for (tokens_array, 0..) |v, i| {
        token_strings[i] = v.string.data;
    }

    var tok = try tokenizer.Tokenizer.init(allocator, token_strings);
    defer tok.deinit(allocator);

    try stdout.interface.print("Tokenizer initialized with {d} tokens.\n", .{tokens_array.len});

    const kernels_source = @embedFile("metal/kernels.metal");
    const dev = try metal.Device.init(kernels_source);
    defer dev.deinit();

    try stdout.interface.print("Metal Device: {s}\n", .{dev.getName()});

    const e = try engine.Engine.init(allocator, dev, model, buffer);
    defer e.deinit();

    try stdout.interface.print("Engine initialized with {d} tensors.\n", .{e.tensors.count()});

    if (prompt) |p| {
        try stdout.interface.print("\nPrompt: {s}\n", .{p});
        const ids = try tok.encode(allocator, p);
        defer allocator.free(ids);

        try stdout.interface.print("zLLM: ", .{});
        try stdout.flush();

        var pos: u32 = 0;
        var current_token: u32 = 0;

        // Prefill
        for (ids) |token| {
            const logits = try e.forward(token, pos);
            current_token = sampler.sampleArgmax(logits);
            pos += 1;
        }

        // Generation loop
        var step: usize = 0;
        while (step < 50) : (step += 1) {
            const token_str = tok.decode(current_token);
            try stdout.interface.print("{s}", .{token_str});
            try stdout.flush();

            // Check for EOS (standard Qwen2.5/Llama3 EOS tokens)
            if (current_token == 151643 or current_token == 151645) break;

            const logits = try e.forward(current_token, pos);
            current_token = sampler.sampleArgmax(logits);
            pos += 1;
        }
        try stdout.interface.print("\n", .{});
        try stdout.flush();
        return;
    }

    try stdout.interface.print("\nWelcome to zLLM! Enter your prompt below (type 'exit' to quit).\n", .{});

    var line_buf: [1024]u8 = undefined;
    var in_buf: [1024]u8 = undefined;
    var stdin = std.Io.File.stdin().reader(io, &in_buf);

    while (true) {
        try stdout.interface.print("> ", .{});
        try stdout.flush();

        const line = try readLine(&stdin.interface, &line_buf) orelse break;
        const trimmed = std.mem.trim(u8, line, " \r\n");
        if (trimmed.len == 0) continue;
        if (std.mem.eql(u8, trimmed, "exit") or std.mem.eql(u8, trimmed, "quit")) break;

        const ids = try tok.encode(allocator, trimmed);
        defer allocator.free(ids);

        try stdout.interface.print("zLLM: ", .{});
        try stdout.flush();

        var pos: u32 = 0;
        var current_token: u32 = 0;

        // Prefill
        for (ids) |token| {
            const logits = try e.forward(token, pos);
            current_token = sampler.sampleArgmax(logits);
            pos += 1;
        }

        // Generation loop
        var step: usize = 0;
        while (step < 50) : (step += 1) {
            const token_str = tok.decode(current_token);
            try stdout.interface.print("{s}", .{token_str});
            try stdout.flush();

            if (current_token == 151643 or current_token == 151645) break;

            const logits = try e.forward(current_token, pos);
            current_token = sampler.sampleArgmax(logits);
            pos += 1;
        }
        try stdout.interface.print("\n", .{});
        try stdout.flush();
    }
}

