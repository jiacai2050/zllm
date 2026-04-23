const std = @import("std");
const metal = @import("metal.zig");

pub fn main(init: std.process.Init) !void {
    _ = init;
    std.debug.print("zLLM starting (debug)...\n", .{});

    const dev = try metal.Device.init();
    defer dev.deinit();

    std.debug.print("Metal Device: {s}\n", .{dev.getName()});
}
