package_group(
    name = "friends",
    packages = [
        "//xla/mlir/...",
        # copybara:uncomment_begin(google-only)
        # # TODO(ezhulenev): Clean up dependencies that are leforvers from Autofusion project.
        # "//third_party/tf_runtime/...",
        # copybara:uncomment_end(google-only)
    ],
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)
