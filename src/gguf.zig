const std = @import("std");

pub const Magic = [4]u8;
pub const GGUF_MAGIC: Magic = "GGUF".*;

pub const Header = struct {
    magic: Magic,
    version: u32,
    tensor_count: u64,
    metadata_kv_count: u64,
};

pub fn parseHeader(reader: anytype) !Header {
    var magic: Magic = undefined;
    _ = try reader.readSliceAll(&magic);

    if (!std.mem.eql(u8, &magic, &GGUF_MAGIC)) {
        return error.InvalidMagic;
    }

    const version = try reader.takeInt(u32, .little);
    const tensor_count = try reader.takeInt(u64, .little);
    const metadata_kv_count = try reader.takeInt(u64, .little);

    return Header{
        .magic = magic,
        .version = version,
        .tensor_count = tensor_count,
        .metadata_kv_count = metadata_kv_count,
    };
}
