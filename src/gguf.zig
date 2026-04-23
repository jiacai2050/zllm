const std = @import("std");

pub const Magic = [4]u8;
pub const GGUF_MAGIC: Magic = "GGUF".*;

pub const Model = struct {
    header: Header,
    metadata: []MetadataKV,
    tensors: []TensorInfo,
    data_offset: u64,

    pub fn findMetadata(self: Model, key: []const u8) ?MetadataKV {
        for (self.metadata) |kv| {
            if (std.mem.eql(u8, kv.key.data, key)) {
                return kv;
            }
        }
        return null;
    }

    pub fn deinit(self: Model, allocator: std.mem.Allocator) void {
        for (self.metadata) |kv| {
            kv.deinit(allocator);
        }
        allocator.free(self.metadata);
        for (self.tensors) |t| {
            allocator.free(t.name.data);
            allocator.free(t.dimensions);
        }
        allocator.free(self.tensors);
    }
};

pub fn parse(reader: *std.Io.Reader, allocator: std.mem.Allocator) !Model {
    const header = try parseHeader(reader);

    const metadata = try allocator.alloc(MetadataKV, header.metadata_kv_count);
    errdefer allocator.free(metadata);
    for (0..header.metadata_kv_count) |i| {
        metadata[i] = try MetadataKV.parse(reader, allocator);
    }

    const tensors = try allocator.alloc(TensorInfo, header.tensor_count);
    errdefer allocator.free(tensors);
    for (0..header.tensor_count) |i| {
        tensors[i] = try parseTensorInfo(reader, allocator);
    }

    // Default alignment for tensor data is 32 bytes.
    var alignment: u64 = 32;
    for (metadata) |kv| {
        if (std.mem.eql(u8, kv.key.data, "general.alignment")) {
            if (kv.value == .uint32) {
                alignment = kv.value.uint32;
            } else if (kv.value == .uint64) {
                alignment = kv.value.uint64;
            }
        }
    }

    // Ensure the alignment is a power of two for efficient bitwise operations.
    std.debug.assert(alignment > 0);
    std.debug.assert((alignment & (alignment - 1)) == 0);

    // Calculate the data offset from the current reader position, aligned to the specified alignment.
    const current_position = reader.seek;
    const data_offset = (current_position + alignment - 1) & ~(alignment - 1);

    return Model{
        .header = header,
        .metadata = metadata,
        .tensors = tensors,
        .data_offset = data_offset,
    };
}

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

pub const Value = union(ValueType) {
    uint8: u8,
    iint8: i8,
    uint16: u16,
    iint16: i16,
    uint32: u32,
    iint32: i32,
    float32: f32,
    bool: bool,
    string: String,
    array: Array,
    uint64: u64,
    iint64: i64,
    float64: f64,

    pub const Array = struct {
        type: ValueType,
        len: u64,
        data: []Value,
    };

    pub fn parse(reader: anytype, allocator: std.mem.Allocator, val_type: ValueType) !Value {
        switch (val_type) {
            .uint8 => return Value{ .uint8 = try reader.takeInt(u8, .little) },
            .iint8 => return Value{ .iint8 = try reader.takeInt(i8, .little) },
            .uint16 => return Value{ .uint16 = try reader.takeInt(u16, .little) },
            .iint16 => return Value{ .iint16 = try reader.takeInt(i16, .little) },
            .uint32 => return Value{ .uint32 = try reader.takeInt(u32, .little) },
            .iint32 => return Value{ .iint32 = try reader.takeInt(i32, .little) },
            .float32 => {
                const bits = try reader.takeInt(u32, .little);
                return Value{ .float32 = @bitCast(bits) };
            },
            .bool => return Value{ .bool = (try reader.takeInt(u8, .little)) != 0 },
            .string => return Value{ .string = try String.parse(reader, allocator) },
            .array => {
                const array_type = try reader.takeEnum(ValueType, .little);
                const len = try reader.takeInt(u64, .little);
                const data = try allocator.alloc(Value, len);
                errdefer allocator.free(data);
                for (0..len) |i| {
                    data[i] = try Value.parse(reader, allocator, array_type);
                }
                return Value{ .array = .{ .type = array_type, .len = len, .data = data } };
            },
            .uint64 => return Value{ .uint64 = try reader.takeInt(u64, .little) },
            .iint64 => return Value{ .iint64 = try reader.takeInt(i64, .little) },
            .float64 => {
                const bits = try reader.takeInt(u64, .little);
                return Value{ .float64 = @bitCast(bits) };
            },
        }
    }

    pub fn deinit(self: Value, allocator: std.mem.Allocator) void {
        switch (self) {
            .string => |s| allocator.free(s.data),
            .array => |a| {
                for (a.data) |v| {
                    v.deinit(allocator);
                }
                allocator.free(a.data);
            },
            else => {},
        }
    }
};

pub const MetadataKV = struct {
    key: String,
    value: Value,

    pub fn parse(reader: anytype, allocator: std.mem.Allocator) !MetadataKV {
        const key = try String.parse(reader, allocator);
        errdefer allocator.free(key.data);
        const val_type = try reader.takeEnum(ValueType, .little);
        const value = try Value.parse(reader, allocator, val_type);
        return MetadataKV{ .key = key, .value = value };
    }

    pub fn deinit(self: MetadataKV, allocator: std.mem.Allocator) void {
        allocator.free(self.key.data);
        self.value.deinit(allocator);
    }
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
