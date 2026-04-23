const std = @import("std");
const gguf = @import("gguf");
const testing = std.testing;

test "parseHeader" {
    var buffer: [32]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buffer);

    // Magic: "GGUF"
    try writer.writeAll("GGUF");
    // Version: 3 (little endian)
    try writer.writeInt(u32, 3, .little);
    // Tensor count: 10
    try writer.writeInt(u64, 10, .little);
    // Metadata KV count: 5
    try writer.writeInt(u64, 5, .little);

    const reader: std.Io.Reader = .fixed(buffer[0..writer.end]);
    const header = try gguf.parseHeader(@constCast(&reader));

    try testing.expectEqual(gguf.GGUF_MAGIC, header.magic);
    try testing.expectEqual(@as(u32, 3), header.version);
    try testing.expectEqual(@as(u64, 10), header.tensor_count);
    try testing.expectEqual(@as(u64, 5), header.metadata_kv_count);
}
