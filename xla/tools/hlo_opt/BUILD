load("//xla:glob_lit_test.bzl", "glob_lit_tests")
load(
    "//xla/stream_executor:build_defs.bzl",
    "if_gpu_is_configured",
)
load("@tsl//tsl:tsl.default.bzl", "filegroup")
load(
    "@tsl//tsl/platform:build_config_root.bzl",
    "tf_cuda_tests_tags",
)
load("@tsl//tsl/platform:rules_cc.bzl", "cc_library")

# hlo-opt tool.
load(
    "@tsl//tsl/platform/default:cuda_build_defs.bzl",
    "if_cuda_is_configured",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//xla:internal"],
    licenses = ["notice"],
)

# Includes a macro to register a provider.
cc_library(
    name = "opt_lib",
    srcs = ["opt_lib.cc"],
    hdrs = ["opt_lib.h"],
    deps = [
        "//xla:debug_options_flags",
        "//xla:statusor",
        "//xla:types",
        "//xla/hlo/ir:hlo",
        "//xla/service:compiler",
        "//xla/stream_executor:platform",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "gpu_opt",
    testonly = True,
    srcs = if_cuda_is_configured(["gpu_opt.cc"]),
    deps = [
        ":opt_lib",
        "//xla:debug_options_flags",
        "//xla:statusor",
        "//xla:types",
        "//xla/service:compiler",
        "//xla/service:dump",
        "//xla/service:platform_util",
        "//xla/service/gpu:executable_proto_cc",
        "//xla/stream_executor/cuda:cuda_platform_id",
        "//xla/stream_executor/platform",
        "@com_google_absl//absl/container:flat_hash_map",
    ] + if_gpu_is_configured([
        "//xla/service:gpu_plugin",
        "//xla/service/gpu:gpu_executable",
    ]) + if_cuda_is_configured([
        "//xla/stream_executor:cuda_platform",
    ]),
    alwayslink = True,  # Initializer needs to run.
)

cc_library(
    name = "opt_main",
    testonly = True,
    srcs = ["opt_main.cc"],
    deps = [
        ":opt_lib",
        "//xla:debug_options_flags",
        "//xla:status",
        "//xla:statusor",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_runner",
        "//xla/service:platform_util",
        "//xla/tools:hlo_module_loader",
        "//xla/tools:run_hlo_module_lib",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:path",
        "@tsl//tsl/platform:platform_port",
        "@tsl//tsl/platform:status",
        "@tsl//tsl/platform:test",
        "@tsl//tsl/util:command_line_flags",
    ] + if_gpu_is_configured([
        ":gpu_opt",
    ]) + if_cuda_is_configured([
        "//xla/stream_executor:cuda_platform",
    ]),
)

glob_lit_tests(
    name = "gpu_opt_tests",
    data = [":test_utilities"],
    default_tags = tf_cuda_tests_tags() + [
    ],
    driver = "//xla:run_lit.sh",
    test_file_exts = ["hlo"],
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        "gpu_specs/a100.txtpb",
        "//xla/tools:hlo-opt",
        "@llvm-project//llvm:FileCheck",
        "@llvm-project//mlir:run_lit.sh",
    ],
)
