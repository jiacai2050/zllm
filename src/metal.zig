const std = @import("std");
const testing = std.testing;

pub const Device = opaque {
    extern fn zllm_metal_init(shader_source: [*:0]const u8) ?*Device;
    extern fn zllm_metal_get_device_name(device: *Device) [*:0]const u8;
    extern fn zllm_metal_create_buffer(device: *Device, data: ?*const anyopaque, size: usize) ?*Buffer;
    extern fn zllm_metal_dispatch(
        device: *Device,
        kernel_name: [*:0]const u8,
        buffers: [*]const ?*Buffer,
        n_buffers: i32,
        threads: i32,
    ) void;
    extern fn zllm_metal_deinit(device: *Device) void;

    /// Initialize the default Metal device with shader source.
    pub fn init(shader_source: [:0]const u8) !*Device {
        return zllm_metal_init(shader_source.ptr) orelse error.MetalInitializationFailed;
    }

    /// Get the name of the Metal device.
    pub fn getName(self: *Device) []const u8 {
        const name_ptr = zllm_metal_get_device_name(self);
        return std.mem.span(name_ptr);
    }

    /// Create a Metal buffer.
    pub fn createBuffer(self: *Device, data: ?[]const u8, size: usize) !*Buffer {
        const data_ptr = if (data) |d| d.ptr else null;
        return zllm_metal_create_buffer(self, data_ptr, size) orelse error.BufferCreationFailed;
    }

    /// Dispatch a compute kernel.
    pub fn dispatch(
        self: *Device,
        kernel_name: [:0]const u8,
        buffers: []const ?*Buffer,
        threads: i32,
    ) void {
        zllm_metal_dispatch(self, kernel_name.ptr, buffers.ptr, @intCast(buffers.len), threads);
    }

    /// Deinitialize the Metal device and free associated resources.
    pub fn deinit(self: *Device) void {
        zllm_metal_deinit(self);
    }
};

pub const Buffer = opaque {
    extern fn zllm_metal_release_buffer(buffer: *Buffer) void;

    pub fn release(self: *Buffer) void {
        zllm_metal_release_buffer(self);
    }
};

test "Metal device initialization" {
    // Initialize the device with a dummy shader.
    const dev = try Device.init("kernel void dummy() {}");
    defer dev.deinit();

    // Verify the device name is not empty.
    const name = dev.getName();
    try testing.expect(name.len > 0);

    // Print the device name for manual verification.
    std.debug.print("Metal device initialized: {s}\n", .{name});
}
