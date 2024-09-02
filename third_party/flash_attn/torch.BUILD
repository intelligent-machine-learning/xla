package(
    default_visibility = [
        "@flash_attn//:__pkg__",
    ],
)

cc_library(
    name = "headers",
    hdrs = glob(
        [
            "torch/include/**/*.cuh",
            "torch/include/**/*.h",
        ],
        ["torch/include/google/protobuf/**/*.h"],
    ),
    strip_include_prefix = "torch/include",
)

cc_import(
    name = "libtorch_base",
    shared_library = "build/lib/libtorch.so",
)

cc_import(
    name = "libtorch_cuda",
    shared_library = "build/lib/libtorch_cuda.so",
)

cc_library(
    name = "libtorch",
    deps = [
        ":libtorch_base",
        ":libtorch_cuda",
    ],
)

cc_import(
    name = "libc10_base",
    shared_library = "build/lib/libc10.so",
)

cc_import(
    name = "libc10_cuda",
    shared_library = "build/lib/libc10_cuda.so",
)

cc_library(
    name = "libc10",
    deps = [
        ":libc10_base",
        ":libc10_cuda",
    ],
)
