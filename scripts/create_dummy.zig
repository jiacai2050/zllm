const std = @import("std");

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const file = try std.Io.Dir.cwd().createFile(io, "dummy.gguf", .{});
    defer file.close(io);

    try file.writeStreamingAll(io, "GGUF");
    
    var buf4: [4]u8 = undefined;
    var buf8: [8]u8 = undefined;

    std.mem.writeInt(u32, &buf4, 3, .little);
    try file.writeStreamingAll(io, &buf4);

    std.mem.writeInt(u64, &buf8, 1, .little); // 1 tensor
    try file.writeStreamingAll(io, &buf8);

    std.mem.writeInt(u64, &buf8, 0, .little); // 0 kv
    try file.writeStreamingAll(io, &buf8);

    // Write 1 TensorInfo
    const name = "token_embd.weight";
    std.mem.writeInt(u64, &buf8, name.len, .little);
    try file.writeStreamingAll(io, &buf8);
    try file.writeStreamingAll(io, name);

    std.mem.writeInt(u32, &buf4, 2, .little); // 2 dims
    try file.writeStreamingAll(io, &buf4);
    std.mem.writeInt(u64, &buf8, 10, .little); // dim0
    try file.writeStreamingAll(io, &buf8);
    std.mem.writeInt(u64, &buf8, 4, .little); // dim1
    try file.writeStreamingAll(io, &buf8);

    std.mem.writeInt(u32, &buf4, 6, .little); // type float32
    try file.writeStreamingAll(io, &buf4);
    std.mem.writeInt(u64, &buf8, 0, .little); // offset 0 (from data_offset)
    try file.writeStreamingAll(io, &buf8);

    // Padding to 32 bytes alignment
    // Header (24) + TensorInfo (Name len 8 + Name 17 + Dims count 4 + Dim0 8 + Dim1 8 + Type 4 + Offset 8 = 57)
    // Total header size = 24 + 57 = 81.
    // Next 32-byte alignment is 96.
    const padding = 96 - 81;
    var pad: [32]u8 = [_]u8{0} ** 32;
    try file.writeStreamingAll(io, pad[0..padding]);

    // Data: 10 * 4 * 4 = 160 bytes
    var data: [160]u8 = [_]u8{0} ** 160;
    try file.writeStreamingAll(io, &data);
}
