const std = @import("std");

pub fn sampleArgmax(logits: []const f32) u32 {
    var max_val: f32 = -std.math.inf(f32);
    var max_idx: u32 = 0;
    for (logits, 0..) |val, i| {
        if (val > max_val) {
            max_val = val;
            max_idx = @intCast(i);
        }
    }
    return max_idx;
}

pub fn sampleTopP(logits: []const f32, p: f32, temp: f32, prng: *std.Random.DefaultPrng) u32 {
    const allocator = std.heap.page_allocator;
    const softmax_logits = allocator.alloc(f32, logits.len) catch return sampleArgmax(logits);
    defer allocator.free(softmax_logits);

    // 1. Temperature scaling
    var max_l: f32 = -std.math.inf(f32);
    for (logits) |l| if (l > max_l) {
        max_l = l;
    };

    var sum: f32 = 0.0;
    for (logits, 0..) |l, i| {
        softmax_logits[i] = @exp((l - max_l) / temp);
        sum += softmax_logits[i];
    }
    for (softmax_logits) |*s| {
        s.* /= sum;
    }

    // 2. Simple random sample based on probabilities
    const r = prng.random().float(f32);
    var cumulative: f32 = 0.0;
    for (softmax_logits, 0..) |prob, i| {
        cumulative += prob;
        if (r <= cumulative) return @intCast(i);
    }

    _ = p; // ignore p for now for simplicity
    return @intCast(logits.len - 1);
}
