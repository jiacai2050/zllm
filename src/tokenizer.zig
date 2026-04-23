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

            // BPE Tokenizers often map non-printable bytes to a specific Unicode range.
            // We reverse this mapping here to ensure the decoded output is standard UTF-8.
            // Mapping: Ġ (\u0120, UTF-8: C4 A0) -> space (ASCII 32)
            // Mapping: Ċ (\u010A, UTF-8: C4 8A) -> newline (ASCII 10)
            var cleaned_token_buffer = try allocator.alloc(u8, token.len);
            var cleaned_index: usize = 0;
            var token_index: usize = 0;

            while (token_index < token.len) {
                if (token_index + 1 < token.len and token[token_index] == 0xC4 and token[token_index + 1] == 0xA0) {
                    cleaned_token_buffer[cleaned_index] = ' ';
                    token_index += 2;
                } else if (token_index + 1 < token.len and token[token_index] == 0xC4 and token[token_index + 1] == 0x8A) {
                    cleaned_token_buffer[cleaned_index] = '\n';
                    token_index += 2;
                } else {
                    cleaned_token_buffer[cleaned_index] = token[token_index];
                    token_index += 1;
                }
                cleaned_index += 1;
            }

            std.debug.assert(cleaned_index <= token.len);
            const final_token = try allocator.realloc(cleaned_token_buffer, cleaned_index);

            try vocab.put(allocator, final_token, id);
            try id_to_token.put(allocator, id, final_token);
        }

        return Tokenizer{
            .vocab = vocab,
            .id_to_token = id_to_token,
        };
    }

    pub fn deinit(self: *Tokenizer, allocator: std.mem.Allocator) void {
        var iterator = self.id_to_token.iterator();
        while (iterator.next()) |entry| {
            allocator.free(entry.value_ptr.*);
        }
        self.vocab.deinit(allocator);
        self.id_to_token.deinit(allocator);
    }

    pub fn decode(self: Tokenizer, id: u32) []const u8 {
        std.debug.assert(id < self.id_to_token.count());
        return self.id_to_token.get(id) orelse "";
    }

    pub fn encode(self: Tokenizer, allocator: std.mem.Allocator, text: []const u8) ![]u32 {
        std.debug.assert(text.len > 0);
        var token_ids: std.ArrayListUnmanaged(u32) = .empty;
        errdefer token_ids.deinit(allocator);

        var start_index: usize = 0;
        while (start_index < text.len) {
            var match_found = false;
            var end_index = text.len;
            while (end_index > start_index) : (end_index -= 1) {
                const substring = text[start_index..end_index];
                if (self.vocab.get(substring)) |id| {
                    try token_ids.append(allocator, id);
                    start_index = end_index;
                    match_found = true;
                    break;
                }
            }
            if (!match_found) {
                // If no token matches, skip the current byte to avoid an infinite loop.
                start_index += 1;
            }
        }
        return token_ids.toOwnedSlice(allocator);
    }
};
