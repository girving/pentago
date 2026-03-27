# Pentago

workspace(name = "pentago")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//third_party:mpi.bzl", "mpi_repository")
load("//third_party:tensorflow.bzl", "tensorflow_repository")

http_archive(
    name = "random123",
    urls = [
        "https://github.com/DEShawResearch/random123/archive/9545ff6413f258be2f04c1d319d99aaef7521150.tar.gz",
    ],
    sha256 = "306f3bf8d9a11298f77ed9a548f524d135b45a722836889f27e05e1acff24676",
    strip_prefix = "random123-9545ff6413f258be2f04c1d319d99aaef7521150",
    build_file = "//third_party:random123.BUILD",
)

http_archive(
    name = "tinyformat",
    urls = [
        "https://github.com/c42f/tinyformat/archive/3a33bbf65442432277eee079e83d3e8fac51730c.tar.gz",
    ],
    sha256 = "52c7b9cb9558f57fbfbdbcfbb9d956793a475886a0f7a21632115978cdd7f8be",
    strip_prefix = "tinyformat-3a33bbf65442432277eee079e83d3e8fac51730c",
    build_file = "//third_party:tinyformat.BUILD",
)

http_archive(
    name = "com_google_googletest",
    urls = [
        "https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz",
    ],
    sha256 = "7b42b4d6ed48810c5362c265a17faebe90dc2373c885e5216439d37927f02926",
    strip_prefix = "googletest-1.15.2",
)

http_archive(
    name = "lzma",
    urls = [
        "https://github.com/tukaani-project/xz/releases/download/v5.2.13/xz-5.2.13.tar.gz",
    ],
    sha256 = "2942a1a8397cd37688f79df9584947d484dd658db088d51b790317eb3184827b",
    strip_prefix = "xz-5.2.13/src",
    build_file = "//third_party:lzma.BUILD",
)

http_archive(
    name = "snappy",
    urls = ["https://github.com/google/snappy/archive/1.1.7.tar.gz"],
    sha256 = "3dfa02e873ff51a11ee02b9ca391807f0c8ea0529a4924afa645fbf97163f9d4",
    strip_prefix = "snappy-1.1.7",
    build_file = "//third_party:snappy.BUILD",
)

http_archive(
    name = "zlib",
    urls = ["https://www.zlib.net/fossils/zlib-1.3.1.tar.gz"],
    sha256 = "9a93b2b7dfdac77ceba5a558a580e74667dd6fede4585b91eefb60f03b72df23",
    strip_prefix = "zlib-1.3.1",
    build_file = "//third_party:zlib.BUILD",
)

mpi_repository(name = "mpi")
tensorflow_repository(name = "tensorflow")
