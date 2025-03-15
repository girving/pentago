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
        "https://github.com/google/googletest/archive/d175c8bf823e709d570772b038757fadf63bc632.tar.gz",
    ],
    sha256 = "39a708e81cf68af02ca20cad879d1dbd055364f3ae5588a5743c919a51d7ad46",
    strip_prefix = "googletest-d175c8bf823e709d570772b038757fadf63bc632",
    build_file = "//third_party:googletest.BUILD",
)

http_archive(
    name = "lzma",
    urls = [
        "https://tukaani.org/xz/xz-5.2.5.tar.gz",
        "https://fossies.org/linux/misc/xz-5.2.5.tar.gz",
    ],
    sha256 = "f6f4910fd033078738bd82bfba4f49219d03b17eb0794eb91efbae419f4aba10",
    strip_prefix = "xz-5.2.5/src",
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
    urls = ["https://www.zlib.net/fossils/zlib-1.2.12.tar.gz"],
    sha256 = "91844808532e5ce316b3c010929493c0244f3d37593afd6de04f71821d5136d9",
    strip_prefix = "zlib-1.2.12",
    build_file = "//third_party:zlib.BUILD",
)

mpi_repository(name = "mpi")
tensorflow_repository(name = "tensorflow")
