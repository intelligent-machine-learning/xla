load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")


def repo():
    http_archive(
        name="sqlite", build_file="//third_party/sqlite:BUILD",
        url="https://www.sqlite.org/2024/sqlite-autoconf-3450300.tar.gz",
        sha256="b2809ca53124c19c60f42bf627736eae011afdcc205bb48270a5ee9a3819153"
        "1",
        strip_prefix="sqlite-autoconf-3450300")
