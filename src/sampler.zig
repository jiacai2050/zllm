const std = @import("std");

pub fn sampleArgmax(logits: []const f32) u32 {
    var max_i: u32 = 0;
    var max_v: f32 = logits[0];
    for (logits, 0..) |v, i| {
        if (v > max_v) {
            max_v = v;
            max_i = @intCast(i);
        }
    }
    return max_i;
}
