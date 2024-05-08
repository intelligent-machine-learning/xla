"""Provides the repository macro to import flash-attention."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    # v2.5.7
    FLASH_ATTN_COMMIT = "85881f547fd1053a7b4a2c3faad6690cca969279"
    FLASH_ATTN_SHA256 = "66f1c7c09d0783c2b5d89b17b542562166d4276b180ae5cad184ad8f2f32d115"

    tf_http_archive(
        name = "flash_attn",
        sha256 = FLASH_ATTN_SHA256,
        strip_prefix = "flash-attention-{commit}".format(commit = FLASH_ATTN_COMMIT),
        urls = tf_mirror_urls("https://github.com/Dao-AILab/flash-attention/archive/{commit}.tar.gz".format(commit = FLASH_ATTN_COMMIT)),
        build_file = "//third_party/flash_attn:flash_attn.BUILD",
        patch_file = [
            "//third_party/flash_attn:flash_attn.patch"
        ],
    )
