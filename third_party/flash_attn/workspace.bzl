"""Provides the repository macro to import flash-attention."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    # v2.5.9.post1
    FLASH_ATTN_COMMIT = "e2e4333c955b829d0e6087d27ee435f55c80d3a5"
    FLASH_ATTN_SHA256 = "0611ec789aa8b837d88209a51c172fecc1a91b54b6bf5a21070c5867005eb079"

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
