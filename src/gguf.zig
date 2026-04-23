const std = @import("std");

pub const Magic = [4]u8;
pub const GGUF_MAGIC: Magic = "GGUF".*;

pub const Header = struct {
    magic: Magic,
    version: u32,
    tensor_count: u64,
    metadata_kv_count: u64,
};

pub const ValueType = enum(u32) {
    uint8 = 0,
    iint8 = 1,
    uint16 = 2,
    iint16 = 3,
    uint32 = 4,
    iint32 = 5,
    float32 = 6,
    bool = 7,
    string = 8,
    array = 9,
    uint64 = 10,
    iint64 = 11,
    float64 = 12,
};

pub const String = struct {
    len: u64,
    data: []const u8,

    pub fn parse(reader: anytype, allocator: std.mem.Allocator) !String {
        const len = try reader.takeInt(u64, .little);
        const data = try allocator.alloc(u8, len);
        errdefer allocator.free(data);

        _ = try reader.readSliceAll(data);

        return String{
            .len = len,
            .data = data,
        };
    }
};

pub const TensorInfo = struct {
    name: String,
    n_dims: u32,
    dimensions: []u64,
    type: u32,
    offset: u64,
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

pub fn parseTensorInfo(reader: anytype, allocator: std.mem.Allocator) !TensorInfo {
    const name = try String.parse(reader, allocator);
    errdefer allocator.free(name.data);

    const n_dims = try reader.takeInt(u32, .little);
    const dimensions = try allocator.alloc(u64, n_dims);
    errdefer allocator.free(dimensions);

    for (0..n_dims) |i| {
        dimensions[i] = try reader.takeInt(u64, .little);
    }

    const tensor_type = try reader.takeInt(u32, .little);
    const offset = try reader.takeInt(u64, .little);

    return TensorInfo{
        .name = name,
        .n_dims = n_dims,
        .dimensions = dimensions,
        .type = tensor_type,
        .offset = offset,
    };
}
