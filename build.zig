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
    root_module.link_libcpp = true;

    // Link macOS frameworks for Metal acceleration.
    root_module.linkFramework("Foundation", .{});
    root_module.linkFramework("Metal", .{});
    root_module.linkFramework("QuartzCore", .{});

    root_module.addCSourceFile(.{
        .file = b.path("src/metal/bridge.mm"),
        .flags = &.{ "-std=c++17", "-fobjc-arc" },
    });
    root_module.addIncludePath(b.path("src/metal"));

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

    // Add test step
    const metal_tests_module = b.createModule(.{
        .root_source_file = b.path("src/metal.zig"),
        .target = target,
        .optimize = optimize,
    });
    metal_tests_module.link_libc = true;
    metal_tests_module.link_libcpp = true;
    metal_tests_module.linkFramework("Foundation", .{});
    metal_tests_module.linkFramework("Metal", .{});
    metal_tests_module.linkFramework("QuartzCore", .{});

    metal_tests_module.addCSourceFile(.{
        .file = b.path("src/metal/bridge.mm"),
        .flags = &.{ "-std=c++17", "-fobjc-arc" },
    });
    metal_tests_module.addIncludePath(b.path("src/metal"));

    const metal_tests = b.addTest(.{
        .root_module = metal_tests_module,
    });

    const run_metal_tests = b.addRunArtifact(metal_tests);
    const test_step = b.step("test", "Run metal tests");
    test_step.dependOn(&run_metal_tests.step);
}
