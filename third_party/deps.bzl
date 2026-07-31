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

    # esbuild, as a pinned static binary per platform: web/client uses it to
    # bundle and minify js/css with no npm toolchain (and hence no dependabot
    # surface).  The tarballs are plain binary releases; only the one matching
    # the host platform is downloaded.
    esbuild_version = "0.25.5"
    esbuild = [
        ("darwin-arm64", "61a312bcb8249d058639c405cf6378dd3107de5535a9974973c48a6dd0d2d062"),
        ("darwin-x64", "d2890be89bd7e322cd83e1665516d3c71b0ac3952655cb27076eddb621a4af24"),
        ("linux-x64", "95a928d8187c6f7ad632a3f3bbf01f66dfd5b5adb724b8bfeec5fff73ff2d91a"),
    ]
    for platform, sha256 in esbuild:
        http_archive(
            name = "esbuild_" + platform.replace("-", "_"),
            urls = ["https://registry.npmjs.org/@esbuild/%s/-/%s-%s.tgz" % (platform, platform, esbuild_version)],
            sha256 = sha256,
            strip_prefix = "package",
            build_file_content = 'exports_files(["bin/esbuild"], visibility = ["//visibility:public"])',
        )

deps = module_extension(implementation = _deps_impl)
