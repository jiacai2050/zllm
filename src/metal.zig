const std = @import("std");
const testing = std.testing;

pub const Device = opaque {
    extern fn zllm_metal_init() ?*Device;
    extern fn zllm_metal_get_device_name(device: *Device) [*:0]const u8;
    extern fn zllm_metal_deinit(device: *Device) void;

    /// Initialize the default Metal device.
    pub fn init() !*Device {
        return zllm_metal_init() orelse error.MetalInitializationFailed;
    }

    /// Get the name of the Metal device.
    /// The returned slice is valid until the device is deinitialized.
    pub fn getName(self: *Device) []const u8 {
        const name_ptr = zllm_metal_get_device_name(self);
        return std.mem.span(name_ptr);
    }

    /// Deinitialize the Metal device and free associated resources.
    pub fn deinit(self: *Device) void {
        zllm_metal_deinit(self);
    }
};

test "Metal device initialization" {
    // Initialize the device.
    const dev = try Device.init();
    defer dev.deinit();

    // Verify the device name is not empty.
    const name = dev.getName();
    try testing.expect(name.len > 0);

    // Print the device name for manual verification.
    std.debug.print("Metal device initialized: {s}\n", .{name});
}
