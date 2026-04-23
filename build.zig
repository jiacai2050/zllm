const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const root_module = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    const exe = b.addExecutable(.{
        .name = "zllm",
        .root_module = root_module,
    });

    // Link LibC as required.
    root_module.link_libc = true;

    // Link macOS frameworks for Metal acceleration.
    root_module.linkFramework("Foundation", .{});
    root_module.linkFramework("Metal", .{});
    root_module.linkFramework("QuartzCore", .{});

    // Install the executable.
    b.installArtifact(exe);

    // Provide a "run" step.
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the application");
    run_step.dependOn(&run_cmd.step);
}
