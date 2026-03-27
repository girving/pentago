"""Module extension for non-BCR http_archive dependencies."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def _deps_impl(ctx):
    http_archive(
        name = "random123",
        urls = [
            "https://github.com/DEShawResearch/random123/archive/9545ff6413f258be2f04c1d319d99aaef7521150.tar.gz",
        ],
        sha256 = "306f3bf8d9a11298f77ed9a548f524d135b45a722836889f27e05e1acff24676",
        strip_prefix = "random123-9545ff6413f258be2f04c1d319d99aaef7521150",
        build_file = Label("//third_party:random123.BUILD"),
    )
    http_archive(
        name = "tinyformat",
        urls = [
            "https://github.com/c42f/tinyformat/archive/3a33bbf65442432277eee079e83d3e8fac51730c.tar.gz",
        ],
        sha256 = "52c7b9cb9558f57fbfbdbcfbb9d956793a475886a0f7a21632115978cdd7f8be",
        strip_prefix = "tinyformat-3a33bbf65442432277eee079e83d3e8fac51730c",
        build_file = Label("//third_party:tinyformat.BUILD"),
    )

deps = module_extension(implementation = _deps_impl)
