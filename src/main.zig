const std = @import("std");
const metal = @import("metal.zig");
const gguf = @import("gguf.zig");
const engine = @import("engine.zig");
const tokenizer = @import("tokenizer.zig");
const sampler = @import("sampler.zig");

/// Read a single line from the standard input until a newline character is encountered.
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

/// Execute the inference process for a given prompt and print the generated text and performance stats.
fn runInference(
    io: std.Io,
    allocator: std.mem.Allocator,
    e: *engine.Engine,
    tok: *tokenizer.Tokenizer,
    stdout: anytype,
    prng: *std.Random.DefaultPrng,
    prompt_text: []const u8,
) !void {
    const token_ids = try tok.encode(allocator, prompt_text);
    defer allocator.free(token_ids);

    try stdout.print("zLLM: ", .{});
    try stdout.flush();

    var current_position: u32 = 0;
    var current_token_id: u32 = 0;

    // Prefill phase: Process all tokens in the prompt.
    for (token_ids) |token_id| {
        const logits = try e.forward(token_id, current_position);
        current_token_id = sampler.sampleTopP(logits, 0.9, 0.8, prng);
        current_position += 1;
    }

    // Generation phase: Predict the next tokens until the limit or EOS is reached.
    const start_time_ns = std.Io.Clock.now(.awake, io).nanoseconds;
    var generated_tokens_count: usize = 0;
    while (generated_tokens_count < 150) {
        const token_string = tok.decode(current_token_id);
        try stdout.print("{s}", .{token_string});
        try stdout.flush();

        // Standard Qwen2.5/Llama3 EOS tokens.
        if (current_token_id == 151643 or current_token_id == 151645) break;

        const logits = try e.forward(current_token_id, current_position);
        current_token_id = sampler.sampleTopP(logits, 0.9, 0.8, prng);
        current_position += 1;
        generated_tokens_count += 1;
    }
    const end_time_ns = std.Io.Clock.now(.awake, io).nanoseconds;

    // Performance metrics calculation.
    const duration_ns = @as(u64, @intCast(end_time_ns - start_time_ns));
    const duration_ms = duration_ns / 1_000_000;
    const tokens_per_second = if (duration_ms > 0) (@as(f64, @floatFromInt(generated_tokens_count)) / @as(f64, @floatFromInt(duration_ms)) * 1000.0) else 0;

    try stdout.print("\n\n--- Stats ---\n", .{});
    try stdout.print("Input tokens:  {d}\n", .{token_ids.len});
    try stdout.print("Output tokens: {d}\n", .{generated_tokens_count});
    try stdout.print("Duration:      {d}ms\n", .{duration_ms});
    try stdout.print("Speed:         {d:.2} tokens/s\n", .{tokens_per_second});
    try stdout.flush();
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const allocator = init.gpa;

    var args_iterator = init.minimal.args.iterate();
    _ = args_iterator.next(); // Skip the program name.
    const model_path = args_iterator.next() orelse {
        std.debug.print("Usage: zllm <model_path> [-p prompt]\n", .{});
        return;
    };

    var prompt_text_arg: ?[]const u8 = null;
    while (args_iterator.next()) |arg| {
        if (std.mem.eql(u8, arg, "-p") or std.mem.eql(u8, arg, "--prompt")) {
            prompt_text_arg = args_iterator.next();
        }
    }

    var output_buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &output_buffer);
    var stdout = &stdout_writer.interface;

    try stdout.print("--- zLLM Inference Engine ---\n", .{});
    try stdout.print("Loading model: {s}\n", .{model_path});

    const file = try std.Io.Dir.cwd().openFile(io, model_path, .{});
    defer file.close(io);

    const stat = try file.stat(io);
    const model_file_size = stat.size;

    const mmap_buffer = try std.posix.mmap(
        null,
        model_file_size,
        .{ .READ = true },
        .{ .TYPE = .SHARED },
        file.handle,
        0,
    );
    defer std.posix.munmap(mmap_buffer);

    var reader: std.Io.Reader = .fixed(mmap_buffer);
    const model = try gguf.parse(&reader, allocator);
    defer model.deinit(allocator);

    var raw_tokens_list = try std.ArrayList([]const u8).initCapacity(allocator, 151936);
    defer raw_tokens_list.deinit(allocator);
    for (model.metadata) |kv| {
        if (std.mem.eql(u8, kv.key.data, "tokenizer.ggml.tokens")) {
            for (kv.value.array.data) |v| {
                try raw_tokens_list.append(allocator, v.string.data);
            }
            break;
        }
    }
    var tok = try tokenizer.Tokenizer.init(allocator, raw_tokens_list.items);
    defer tok.deinit(allocator);

    try stdout.print("GGUF version: {d}\n", .{model.header.version});
    try stdout.print("Tensor count: {d}\n", .{model.header.tensor_count});
    try stdout.print("Tokenizer initialized with {d} tokens.\n", .{raw_tokens_list.items.len});

    const kernels_source = @embedFile("metal/kernels.metal");
    const dev = try metal.Device.init(kernels_source);
    defer dev.deinit();

    try stdout.print("Metal Device: {s}\n", .{dev.getName()});

    const e = try engine.Engine.init(allocator, dev, model, mmap_buffer);
    defer e.deinit();

    try stdout.print("Engine initialized with {d} tensors.\n", .{e.tensors.count()});

    var prng = std.Random.DefaultPrng.init(@intCast(std.Io.Clock.now(.awake, io).nanoseconds));

    if (prompt_text_arg) |p| {
        try stdout.print("\nPrompt: {s}\n", .{p});
        try runInference(io, allocator, e, &tok, stdout, &prng, p);
        return;
    }

    try stdout.print("\nWelcome to zLLM! Enter your prompt below (type 'exit' to quit).\n", .{});

    var line_buffer: [2048]u8 = undefined;
    var input_buffer: [1024]u8 = undefined;
    var stdin = std.Io.File.stdin().reader(io, &input_buffer);

    while (true) {
        try stdout.print("> ", .{});
        try stdout.flush();

        const line = try readLine(&stdin.interface, &line_buffer) orelse break;
        const trimmed_line = std.mem.trim(u8, line, " \r\n");
        if (trimmed_line.len == 0) continue;
        if (std.mem.eql(u8, trimmed_line, "exit") or std.mem.eql(u8, trimmed_line, "quit")) break;

        try runInference(io, allocator, e, &tok, stdout, &prng, trimmed_line);
    }
}
