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

test "String.parse" {
    var buffer: [32]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buffer);

    const test_str = "hello gguf";
    try writer.writeInt(u64, test_str.len, .little);
    try writer.writeAll(test_str);

    var reader: std.Io.Reader = .fixed(buffer[0..writer.end]);
    const str = try gguf.String.parse(&reader, testing.allocator);
    defer testing.allocator.free(str.data);

    try testing.expectEqual(test_str.len, str.len);
    try testing.expectEqualSlices(u8, test_str, str.data);
}

test "parseTensorInfo" {
    var buffer: [128]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buffer);

    const name = "tensor.weight";
    try writer.writeInt(u64, name.len, .little);
    try writer.writeAll(name);

    // n_dims: 2
    try writer.writeInt(u32, 2, .little);
    // dims: [1024, 512]
    try writer.writeInt(u64, 1024, .little);
    try writer.writeInt(u64, 512, .little);
    // type: float32 (6)
    try writer.writeInt(u32, 6, .little);
    // offset: 1234
    try writer.writeInt(u64, 1234, .little);

    var reader: std.Io.Reader = .fixed(buffer[0..writer.end]);
    const info = try gguf.parseTensorInfo(&reader, testing.allocator);
    defer testing.allocator.free(info.name.data);
    defer testing.allocator.free(info.dimensions);

    try testing.expectEqualSlices(u8, name, info.name.data);
    try testing.expectEqual(@as(u32, 2), info.n_dims);
    try testing.expectEqual(@as(u64, 1024), info.dimensions[0]);
    try testing.expectEqual(@as(u64, 512), info.dimensions[1]);
    try testing.expectEqual(@as(u32, 6), info.type);
    try testing.expectEqual(@as(u64, 1234), info.offset);
}
