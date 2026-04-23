const std = @import("std");
const Tokenizer = @import("tokenizer").Tokenizer;

test "tokenizer init and decode" {
    std.debug.print("Running tokenizer init and decode test...\n", .{});
    const allocator = std.testing.allocator;
    const tokens = &[_][]const u8{ "hello", "world", "!" };

    var tokenizer = try Tokenizer.init(allocator, tokens);
    defer tokenizer.deinit(allocator);

    try std.testing.expectEqual(@as(u32, 3), tokenizer.vocab.count());
    try std.testing.expectEqualStrings("hello", tokenizer.decode(0));
    try std.testing.expectEqualStrings("world", tokenizer.decode(1));
    try std.testing.expectEqualStrings("!", tokenizer.decode(2));
}

test "tokenizer encode greedy" {
    const allocator = std.testing.allocator;
    const tokens = &[_][]const u8{ "hello", "world", "!" };

    var tokenizer = try Tokenizer.init(allocator, tokens);
    defer tokenizer.deinit(allocator);

    const ids = try tokenizer.encode(allocator, "helloworld!");
    defer allocator.free(ids);

    try std.testing.expectEqual(@as(usize, 3), ids.len);
    try std.testing.expectEqual(@as(u32, 0), ids[0]); // hello
    try std.testing.expectEqual(@as(u32, 1), ids[1]); // world
    try std.testing.expectEqual(@as(u32, 2), ids[2]); // !
}
