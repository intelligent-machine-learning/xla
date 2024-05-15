load("@bazel_skylib//rules:build_test.bzl", "build_test")
load("@rules_proto//proto:defs.bzl", "proto_library")
load("//xla:xla.bzl",  "xla_cc_test","xla_cc_binary")
load("@tsl//tsl:tsl.default.bzl", "get_compatible_with_libtpu_portable")
load("@tsl//tsl:tsl.bzl", "internal_visibility")
load(
    "@local_config_cuda//cuda:build_defs.bzl",
    "if_cuda",
)
load(
    "@tsl//tsl/platform:build_config.bzl",
    "tf_proto_library",
)
package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":friends"],
)

package_group(
    name = "friends",
    packages= [
        # "//xla/hlo/experimental/auto_reorder",
        "//xla/service/gpu/...",
        "//xla/hlo/utils/...",
    ],
    
)

# Filegroup used to collect source files for dependency checking.
filegroup(
    name = "c_srcs",
    data = glob([
        "**/*.cc",
        "**/*.h",
    ]),
)
cc_library(
    name = "auto_reorder_solver",
    srcs = ["auto_reorder_solver.cc",
    ],
    hdrs = [
        "auto_reorder_solver.h",
    ],
    deps = [
        "//xla:statusor",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_parser",
        "//xla/hlo/utils:common_ortools_deps",
        "@tsl//tsl/platform:statusor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@tsl//tsl/platform:hash",
        "@tsl//tsl/platform:types",
    ]
)
# All header files that are used in the build must be declared in
#   the hdrs or srcs of cc_* rules.
#   This is enforced.

cc_library(
    name = "auto_reorder",
    srcs = [
        "auto_reorder.cc",
    ],
    hdrs = [
        "auto_reorder.h",
        "auto_reorder_solver.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_parser",
        "//xla/hlo/ir:hlo_module_group",
        "//xla/service:hlo_pass",
        "//xla/service:hlo_cost_analysis",
        "//xla/service:latency_hiding_scheduler",
        "//xla/service/gpu/model:gpu_hlo_cost_analysis",
        "//xla/service/gpu/model:analytical_latency_estimator",
        "//xla/service:backend",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:statusor",
        ":auto_reorder_solver"
    ],
)
cc_library(
    name="convert_xplane",
    srcs=["convert_xplane.cc"],
    hdrs=["convert_xplane.h"],
    deps=[
        "//xla:status",
        "//xla:xla_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_proto_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:types",
        "@tsl//tsl/profiler/convert:xla_op_utils",
        "@tsl//tsl/profiler/protobuf:profiled_instructions_proto_cc",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@tsl//tsl/profiler/utils:file_system_utils",
        "@tsl//tsl/profiler/utils:tf_xplane_visitor",
        "@tsl//tsl/profiler/utils:xplane_schema",
        "@tsl//tsl/profiler/utils:xplane_utils",
        "@tsl//tsl/profiler/utils:xplane_visitor",
    ]
)
xla_cc_test(
    name = "auto_reorder_test",
    srcs = ["auto_reorder_test.cc"],
    deps = [
        ":auto_reorder",
        ":convert_xplane",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/service:hlo_parser",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "//xla/service:latency_hiding_scheduler",
        "//xla/service/gpu:gpu_hlo_schedule",
        "//xla/service:gpu_plugin",
        "//xla/service/gpu:gpu_device_info_for_tests",
        "@com_google_absl//absl/log",
        "@com_google_googletest//:gtest",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:statusor",
    ],
)

cc_binary(
    name="convert_xplane_tools",
    srcs=["convert_xplane_bin.cc"],
    deps=[
        ":convert_xplane",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_parser",
        "//xla/service:latency_hiding_scheduler",
        "//xla/service/gpu:gpu_hlo_schedule",
        "//xla/service:gpu_plugin",
        "@com_google_absl//absl/log",
        "@tsl//tsl/platform:statusor",

    ]
)
#     # compatible_with = get_compatible_with_libtpu_portable(),
#     deps=[
#         "//xla:status",
#         "//xla:xla_proto_cc",
#         "//xla/hlo/ir:hlo",
#         "//xla/service:hlo_proto_cc",
#         "@com_google_absl//absl/container:flat_hash_map",
#         "@com_google_absl//absl/status",
#         "@com_google_absl//absl/strings",
#         "@com_google_absl//absl/types:optional",
#         "@tsl//tsl/platform:env",
#         "@tsl//tsl/platform:types",
#         "@tsl//tsl/profiler/convert:xla_op_utils",
#         "@tsl//tsl/profiler/protobuf:profiled_instructions_proto_cc",
#         "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
#         "@tsl//tsl/profiler/utils:file_system_utils",
#         "@tsl//tsl/profiler/utils:tf_xplane_visitor",
#         "@tsl//tsl/profiler/utils:xplane_schema",
#         "@tsl//tsl/profiler/utils:xplane_utils",
#         "@tsl//tsl/profiler/utils:xplane_visitor",
#         "@tsl//tsl/platform:platform_port",
#     ]
# )