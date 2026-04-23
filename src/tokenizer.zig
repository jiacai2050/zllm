const std = @import("std");

pub const Tokenizer = struct {
    vocab: std.StringHashMapUnmanaged(u32) = .empty,
    id_to_token: std.AutoHashMapUnmanaged(u32, []const u8) = .empty,

    pub fn init(allocator: std.mem.Allocator, tokens: []const []const u8) !Tokenizer {
        var vocab: std.StringHashMapUnmanaged(u32) = .empty;
        errdefer vocab.deinit(allocator);

        var id_to_token: std.AutoHashMapUnmanaged(u32, []const u8) = .empty;
        errdefer id_to_token.deinit(allocator);

        try vocab.ensureTotalCapacity(allocator, @intCast(tokens.len));
        try id_to_token.ensureTotalCapacity(allocator, @intCast(tokens.len));

        for (tokens, 0..) |token, i| {
            const id = @as(u32, @intCast(i));
            vocab.putAssumeCapacity(token, id);
            id_to_token.putAssumeCapacity(id, token);
        }

        return Tokenizer{
            .vocab = vocab,
            .id_to_token = id_to_token,
        };
    }

    pub fn deinit(self: *Tokenizer, allocator: std.mem.Allocator) void {
        self.vocab.deinit(allocator);
        self.id_to_token.deinit(allocator);
    }

    pub fn decode(self: Tokenizer, id: u32) []const u8 {
        return self.id_to_token.get(id) orelse "";
    }

    pub fn encode(self: Tokenizer, allocator: std.mem.Allocator, text: []const u8) ![]u32 {
        var ids: std.ArrayListUnmanaged(u32) = .empty;
        errdefer ids.deinit(allocator);

        var i: usize = 0;
        while (i < text.len) {
            var found = false;
            var j = text.len;
            while (j > i) : (j -= 1) {
                const sub = text[i..j];
                if (self.vocab.get(sub)) |id| {
                    try ids.append(allocator, id);
                    i = j;
                    found = true;
                    break;
                }
            }
            if (!found) {
                // Fallback for single byte if not found in vocab
                // Many models have single-byte tokens, but if not, we skip
                i += 1;
            }
        }
        return ids.toOwnedSlice(allocator);
    }
};
