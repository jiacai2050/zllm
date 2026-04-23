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

    const ggml_type = std.enums.fromInt(GGMLType, dtype) orelse {
        std.debug.print("Unknown tensor type: {d}\n", .{dtype});
        return error.UnknownTensorType;
    };

    return switch (ggml_type) {
        .F32 => elements * 4,
        .F16 => elements * 2,
        .Q4_0 => (elements / 32) * 18,
        .Q4_1 => (elements / 32) * 20,
        .Q5_0 => (elements / 32) * 22,
        .Q5_1 => (elements / 32) * 24,
        .Q8_0 => (elements / 32) * 34,
        .Q2_K => (elements / 256) * 160,
        .Q3_K => (elements / 256) * 112,
        .Q4_K => (elements / 256) * 144,
        .Q5_K => (elements / 256) * 176,
        .Q6_K => (elements / 256) * 210,
        else => {
            std.debug.print("Unsupported tensor type: {s} ({d})\n", .{ @tagName(ggml_type), dtype });
            return error.UnsupportedTensorType;
        },
    };
}

fn getMetadataUint32(model: gguf.Model, key: []const u8) !u32 {
    const kv = model.findMetadata(key) orelse {
        std.debug.print("Missing metadata: {s}\n", .{key});
        return error.MissingMetadata;
    };
    return switch (kv.value) {
        .uint32 => |v| v,
        .uint64 => |v| @intCast(v),
        .iint32 => |v| @intCast(v),
        .iint64 => |v| @intCast(v),
        else => error.InvalidMetadataType,
    };
}

fn getMetadataFloat32(model: gguf.Model, key: []const u8) !f32 {
    const kv = model.findMetadata(key) orelse {
        std.debug.print("Missing metadata: {s}\n", .{key});
        return error.MissingMetadata;
    };
    return switch (kv.value) {
        .float32 => |v| v,
        .float64 => |v| @floatCast(v),
        else => error.InvalidMetadataType,
    };
}

pub const Engine = struct {
    device: *metal.Device,
    model: gguf.Model,
    tensors: std.StringArrayHashMapUnmanaged(Tensor),
    allocator: std.mem.Allocator,

    // Hyperparameters
    n_embd: u32,
    n_layer: u32,
    n_ff: u32,
    n_head: u32,
    n_head_kv: u32,
    n_vocab: u32,
    rms_norm_eps: f32,
    rope_freq_base: f32,
    head_dim: u32,
    context_length: u32 = 512,

    // Intermediate buffers
    hidden_states: *metal.Buffer,
    norm_states: *metal.Buffer,
    q: *metal.Buffer,
    k: *metal.Buffer,
    v: *metal.Buffer,
    k_cache: *metal.Buffer,
    v_cache: *metal.Buffer,
    attn_scores: *metal.Buffer,
    mlp_gate: *metal.Buffer,
    mlp_up: *metal.Buffer,
    logits: *metal.Buffer,

    // Scalar/Constant buffers
    token_buf: *metal.Buffer,
    pos_buf: *metal.Buffer,
    eps_buf: *metal.Buffer,
    dim_buf: *metal.Buffer,
    head_dim_buf: *metal.Buffer,
    freq_base_buf: *metal.Buffer,
    ilayer_buf: *metal.Buffer,
    n_head_buf: *metal.Buffer,
    n_head_kv_buf: *metal.Buffer,
    context_len_buf: *metal.Buffer,

    const Tensor = struct {
        buffer: *metal.Buffer,
        type: u32,
    };

    pub fn init(allocator: std.mem.Allocator, device: *metal.Device, model: gguf.Model, mmap_buffer: []const u8) !*Engine {
        var tensors: std.StringArrayHashMapUnmanaged(Tensor) = .empty;
        errdefer {
            var it = tensors.iterator();
            while (it.next()) |entry| {
                entry.value_ptr.buffer.release();
            }
            tensors.deinit(allocator);
        }

        for (model.tensors) |t| {
            const offset = model.data_offset + t.offset;
            const size = try getTensorSize(t.type, t.dimensions);

            if (offset + size > mmap_buffer.len) return error.TensorOutOfBounds;

            const buf = try device.createBuffer(mmap_buffer[offset .. offset + size], size);
            try tensors.put(allocator, t.name.data, .{ .buffer = buf, .type = t.type });
        }

        const architecture_kv = model.findMetadata("general.architecture") orelse {
            std.debug.print("Missing metadata: general.architecture\n", .{});
            return error.MissingMetadata;
        };
        const arch = architecture_kv.value.string.data;
        var prefix_buf: [32]u8 = undefined;
        const prefix = try std.fmt.bufPrint(&prefix_buf, "{s}.", .{arch});

        var key_buf: [128]u8 = undefined;
        const n_embd = try getMetadataUint32(model, try std.fmt.bufPrint(&key_buf, "{s}embedding_length", .{prefix}));
        const n_layer = try getMetadataUint32(model, try std.fmt.bufPrint(&key_buf, "{s}block_count", .{prefix}));
        const n_ff = try getMetadataUint32(model, try std.fmt.bufPrint(&key_buf, "{s}feed_forward_length", .{prefix}));
        const n_head = try getMetadataUint32(model, try std.fmt.bufPrint(&key_buf, "{s}attention.head_count", .{prefix}));
        const n_head_kv = try getMetadataUint32(model, try std.fmt.bufPrint(&key_buf, "{s}attention.head_count_kv", .{prefix}));
        
        var n_vocab: u32 = 0;
        if (model.findMetadata("llama.vocab_size")) |kv| {
            n_vocab = kv.value.uint32;
        } else if (model.findMetadata(try std.fmt.bufPrint(&key_buf, "{s}vocab_size", .{prefix}))) |kv| {
            n_vocab = kv.value.uint32;
        } else if (model.findMetadata("tokenizer.ggml.tokens")) |kv| {
            n_vocab = @intCast(kv.value.array.len);
        } else {
            return error.MissingVocabSize;
        }
        
        const rms_norm_eps = getMetadataFloat32(model, try std.fmt.bufPrint(&key_buf, "{s}attention.layer_norm_rms_epsilon", .{prefix})) catch 1e-6;
        const rope_freq_base = getMetadataFloat32(model, try std.fmt.bufPrint(&key_buf, "{s}rope.freq_base", .{prefix})) catch 10000.0;
        const head_dim = n_embd / n_head;
        const context_len: u32 = 512;

        // Allocate intermediate buffers
        const hidden_states = try device.createBuffer(null, n_embd * @sizeOf(f32));
        const norm_states = try device.createBuffer(null, n_embd * @sizeOf(f32));
        const q = try device.createBuffer(null, n_embd * @sizeOf(f32));
        const k = try device.createBuffer(null, (n_head_kv * head_dim) * @sizeOf(f32));
        const v = try device.createBuffer(null, (n_head_kv * head_dim) * @sizeOf(f32));
        const k_cache = try device.createBuffer(null, n_layer * n_head_kv * head_dim * context_len * @sizeOf(f32));
        const v_cache = try device.createBuffer(null, n_layer * n_head_kv * head_dim * context_len * @sizeOf(f32));
        const attn_scores = try device.createBuffer(null, n_head * context_len * @sizeOf(f32));
        const mlp_gate = try device.createBuffer(null, n_ff * @sizeOf(f32));
        const mlp_up = try device.createBuffer(null, n_ff * @sizeOf(f32));
        const logits = try device.createBuffer(null, n_vocab * @sizeOf(f32));

        // Allocate scalar buffers
        const token_buf = try device.createBuffer(null, @sizeOf(u32));
        const pos_buf = try device.createBuffer(null, @sizeOf(u32));
        const eps_buf = try device.createBuffer(null, @sizeOf(f32));
        const dim_buf = try device.createBuffer(null, @sizeOf(u32));
        const head_dim_buf = try device.createBuffer(null, @sizeOf(u32));
        const freq_base_buf = try device.createBuffer(null, @sizeOf(f32));
        const ilayer_buf = try device.createBuffer(null, @sizeOf(u32));
        const n_head_buf = try device.createBuffer(null, @sizeOf(u32));
        const n_head_kv_buf = try device.createBuffer(null, @sizeOf(u32));
        const context_len_buf = try device.createBuffer(null, @sizeOf(u32));

        const self = try allocator.create(Engine);
        self.* = Engine{
            .device = device,
            .model = model,
            .tensors = tensors,
            .allocator = allocator,
            .n_embd = n_embd,
            .n_layer = n_layer,
            .n_ff = n_ff,
            .n_head = n_head,
            .n_head_kv = n_head_kv,
            .n_vocab = n_vocab,
            .rms_norm_eps = rms_norm_eps,
            .rope_freq_base = rope_freq_base,
            .head_dim = head_dim,
            .hidden_states = hidden_states,
            .norm_states = norm_states,
            .q = q,
            .k = k,
            .v = v,
            .k_cache = k_cache,
            .v_cache = v_cache,
            .attn_scores = attn_scores,
            .mlp_gate = mlp_gate,
            .mlp_up = mlp_up,
            .logits = logits,
            .token_buf = token_buf,
            .pos_buf = pos_buf,
            .eps_buf = eps_buf,
            .dim_buf = dim_buf,
            .head_dim_buf = head_dim_buf,
            .freq_base_buf = freq_base_buf,
            .ilayer_buf = ilayer_buf,
            .n_head_buf = n_head_buf,
            .n_head_kv_buf = n_head_kv_buf,
            .context_len_buf = context_len_buf,
        };
        return self;
    }

    pub fn deinit(self: *Engine) void {
        var it = self.tensors.iterator();
        while (it.next()) |entry| {
            entry.value_ptr.buffer.release();
        }
        self.tensors.deinit(self.allocator);

        self.hidden_states.release();
        self.norm_states.release();
        self.q.release();
        self.k.release();
        self.v.release();
        self.k_cache.release();
        self.v_cache.release();
        self.attn_scores.release();
        self.mlp_gate.release();
        self.mlp_up.release();
        self.logits.release();

        self.token_buf.release();
        self.pos_buf.release();
        self.eps_buf.release();
        self.dim_buf.release();
        self.head_dim_buf.release();
        self.freq_base_buf.release();
        self.ilayer_buf.release();
        self.n_head_buf.release();
        self.n_head_kv_buf.release();
        self.context_len_buf.release();

        self.allocator.destroy(self);
    }

    pub fn forward(self: *Engine, token: u32, pos: u32) ![]const f32 {
        // Update scalar buffers
        self.setBufferValue(u32, self.token_buf, token);
        self.setBufferValue(u32, self.pos_buf, pos);
        self.setBufferValue(f32, self.eps_buf, self.rms_norm_eps);
        self.setBufferValue(u32, self.dim_buf, self.n_embd);
        self.setBufferValue(u32, self.head_dim_buf, self.head_dim);
        self.setBufferValue(f32, self.freq_base_buf, self.rope_freq_base);
        self.setBufferValue(u32, self.n_head_buf, self.n_head);
        self.setBufferValue(u32, self.n_head_kv_buf, self.n_head_kv);
        self.setBufferValue(u32, self.context_len_buf, self.context_length);

        // 1. Embedding
        const token_embd = try self.getTensor("token_embd.weight");
        const embed_kernel = if (token_embd.type == @intFromEnum(GGMLType.Q5_0)) "embed_q5_0" else "embed_f32";
        self.device.dispatch(embed_kernel, &.{ self.token_buf, self.hidden_states, token_embd.buffer, self.dim_buf }, @intCast(self.n_embd));

        // 2. Transformer layers
        for (0..self.n_layer) |i| {
            // a. Attention RMSNorm
            var name_buf: [128]u8 = undefined;
            const attn_norm_name = try std.fmt.bufPrintZ(&name_buf, "blk.{d}.attn_norm.weight", .{i});
            const attn_norm_weight = try self.getTensor(attn_norm_name);
            self.device.dispatch("rms_norm", &.{ self.hidden_states, self.norm_states, attn_norm_weight.buffer, self.eps_buf, self.dim_buf }, 1);

            // b. QKV Projections
            var q_name_buf: [128]u8 = undefined;
            var k_name_buf: [128]u8 = undefined;
            var v_name_buf: [128]u8 = undefined;
            const q_name = try std.fmt.bufPrintZ(&q_name_buf, "blk.{d}.attn_q.weight", .{i});
            const k_name = try std.fmt.bufPrintZ(&k_name_buf, "blk.{d}.attn_k.weight", .{i});
            const v_name = try std.fmt.bufPrintZ(&v_name_buf, "blk.{d}.attn_v.weight", .{i});

            const q_weight = try self.getTensor(q_name);
            const k_weight = try self.getTensor(k_name);
            const v_weight = try self.getTensor(v_name);

            self.setBufferValue(u32, self.dim_buf, self.n_embd);
            try self.dispatchMatMul(q_weight, self.norm_states, self.q, self.n_embd);
            try self.dispatchMatMul(k_weight, self.norm_states, self.k, self.n_head_kv * self.head_dim);
            try self.dispatchMatMul(v_weight, self.norm_states, self.v, self.n_head_kv * self.head_dim);

            // Add Bias
            const q_bias_name = try std.fmt.bufPrintZ(&q_name_buf, "blk.{d}.attn_q.bias", .{i});
            if (self.tensors.get(q_bias_name)) |b| self.device.dispatch("add_bias", &.{ b.buffer, self.q }, @intCast(self.n_embd));
            const k_bias_name = try std.fmt.bufPrintZ(&k_name_buf, "blk.{d}.attn_k.bias", .{i});
            if (self.tensors.get(k_bias_name)) |b| self.device.dispatch("add_bias", &.{ b.buffer, self.k }, @intCast(self.n_head_kv * self.head_dim));
            const v_bias_name = try std.fmt.bufPrintZ(&v_name_buf, "blk.{d}.attn_v.bias", .{i});
            if (self.tensors.get(v_bias_name)) |b| self.device.dispatch("add_bias", &.{ b.buffer, self.v }, @intCast(self.n_head_kv * self.head_dim));

            // c. RoPE
            self.device.dispatch("rope", &.{ self.q, self.pos_buf, self.head_dim_buf, self.freq_base_buf }, @intCast(self.n_head));
            self.device.dispatch("rope", &.{ self.k, self.pos_buf, self.head_dim_buf, self.freq_base_buf }, @intCast(self.n_head_kv));

            // d. Attention
            self.setBufferValue(u32, self.ilayer_buf, @intCast(i));
            self.device.dispatch("update_kv_cache", &.{ self.k, self.v, self.k_cache, self.v_cache, self.pos_buf, self.ilayer_buf, self.n_head_kv_buf, self.head_dim_buf, self.context_len_buf }, @intCast(self.n_head_kv * self.head_dim));
            self.device.dispatch("attention", &.{ self.q, self.k_cache, self.v_cache, self.norm_states, self.pos_buf, self.ilayer_buf, self.n_head_buf, self.n_head_kv_buf, self.head_dim_buf, self.context_len_buf }, @intCast(self.n_head));

            // e. Output Projection
            var attn_out_name_buf: [128]u8 = undefined;
            const attn_out_name = try std.fmt.bufPrintZ(&attn_out_name_buf, "blk.{d}.attn_output.weight", .{i});
            const attn_out_weight = try self.getTensor(attn_out_name);
            self.setBufferValue(u32, self.dim_buf, self.n_embd);
            try self.dispatchMatMul(attn_out_weight, self.norm_states, self.q, self.n_embd);

            // f. Residual Add
            self.device.dispatch("add", &.{ self.q, self.hidden_states }, @intCast(self.n_embd));

            // g. FFN RMSNorm
            var ffn_norm_name_buf: [128]u8 = undefined;
            const ffn_norm_name = try std.fmt.bufPrintZ(&ffn_norm_name_buf, "blk.{d}.ffn_norm.weight", .{i});
            const ffn_norm_weight = try self.getTensor(ffn_norm_name);
            self.device.dispatch("rms_norm", &.{ self.hidden_states, self.norm_states, ffn_norm_weight.buffer, self.eps_buf, self.dim_buf }, 1);

            // h. MLP
            var gate_name_buf: [128]u8 = undefined;
            var up_name_buf: [128]u8 = undefined;
            var down_name_buf: [128]u8 = undefined;
            const gate_name = try std.fmt.bufPrintZ(&gate_name_buf, "blk.{d}.ffn_gate.weight", .{i});
            const up_name = try std.fmt.bufPrintZ(&up_name_buf, "blk.{d}.ffn_up.weight", .{i});
            const down_name = try std.fmt.bufPrintZ(&down_name_buf, "blk.{d}.ffn_down.weight", .{i});

            const gate_weight = try self.getTensor(gate_name);
            const up_weight = try self.getTensor(up_name);
            const down_weight = try self.getTensor(down_name);
            
            self.setBufferValue(u32, self.dim_buf, self.n_embd);
            try self.dispatchMatMul(gate_weight, self.norm_states, self.mlp_gate, self.n_ff);
            try self.dispatchMatMul(up_weight, self.norm_states, self.mlp_up, self.n_ff);
            self.device.dispatch("swiglu", &.{ self.mlp_gate, self.mlp_up, self.mlp_gate }, @intCast(self.n_ff));
            
            self.setBufferValue(u32, self.dim_buf, self.n_ff);
            try self.dispatchMatMul(down_weight, self.mlp_gate, self.norm_states, self.n_embd);
            self.setBufferValue(u32, self.dim_buf, self.n_embd);

            // i. Residual Add
            self.device.dispatch("add", &.{ self.norm_states, self.hidden_states }, @intCast(self.n_embd));
        }

        // 3. Final RMSNorm
        const final_norm_weight = try self.getTensor("output_norm.weight");
        self.device.dispatch("rms_norm", &.{ self.hidden_states, self.norm_states, final_norm_weight.buffer, self.eps_buf, self.dim_buf }, 1);

        // 4. Output Projection (Logits)
        const lm_head = self.tensors.get("output.weight") orelse try self.getTensor("token_embd.weight");
        self.setBufferValue(u32, self.dim_buf, self.n_embd);
        try self.dispatchMatMul(lm_head, self.norm_states, self.logits, self.n_vocab);

        // Read back logits
        const logits_ptr = self.logits.getContents() orelse return error.BufferReadFailed;
        const logits: [*]const f32 = @ptrCast(@alignCast(logits_ptr));
        return logits[0..self.n_vocab];
    }

    fn getTensor(self: *Engine, name: []const u8) !Tensor {
        return self.tensors.get(name) orelse {
            std.debug.print("Tensor not found: {s}\n", .{name});
            return error.TensorNotFound;
        };
    }

    fn dispatchMatMul(self: *Engine, weight: Tensor, input: *metal.Buffer, output: *metal.Buffer, n_out: u32) !void {
        const kernel = switch (std.enums.fromInt(GGMLType, weight.type) orelse .F32) {
            .F32 => "matmul_f32",
            .Q4_0 => "matmul_q4_0",
            .Q5_0 => "matmul_q5_0",
            .Q8_0 => "matmul_q8_0",
            .Q4_K => "matmul_q4_K",
            .Q6_K => "matmul_q6_K",
            else => {
                std.debug.print("MatMul: Unsupported tensor type {d}\n", .{weight.type});
                return error.UnsupportedMatMulType;
            },
        };
        self.device.dispatch(kernel, &.{ weight.buffer, input, output, self.dim_buf }, @intCast(n_out));
    }

    fn setBufferValue(self: *Engine, comptime T: type, buffer: *metal.Buffer, value: T) void {
        _ = self;
        const ptr = buffer.getContents() orelse return;
        const typed_ptr: *T = @ptrCast(@alignCast(ptr));
        typed_ptr.* = value;
    }
};
