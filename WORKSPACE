# Pentago

workspace(name = "pentago")

load("//third_party:mpi.bzl", "mpi_repository")

new_http_archive(
    name = "random123",
    urls = [
        "https://github.com/girving/random123/archive/35ff9a55323e6de5d4284895f44721366e535e55.tar.gz",
    ],
    sha256 = "9ee9db7bef212bd050bcdf22cf0f282dd9eccfeda25f35d8790380049bdeead8",
    strip_prefix = "random123-35ff9a55323e6de5d4284895f44721366e535e55",
    build_file = "//third_party:random123.BUILD",
)

new_http_archive(
    name = "tinyformat",
    urls = [
        "https://github.com/c42f/tinyformat/archive/3a33bbf65442432277eee079e83d3e8fac51730c.tar.gz",
    ],
    sha256 = "52c7b9cb9558f57fbfbdbcfbb9d956793a475886a0f7a21632115978cdd7f8be",
    strip_prefix = "tinyformat-3a33bbf65442432277eee079e83d3e8fac51730c",
    build_file = "//third_party:tinyformat.BUILD",
)

new_http_archive(
    name = "com_google_googletest",
    urls = [
        "https://github.com/google/googletest/archive/d175c8bf823e709d570772b038757fadf63bc632.tar.gz",
    ],
    sha256 = "39a708e81cf68af02ca20cad879d1dbd055364f3ae5588a5743c919a51d7ad46",
    strip_prefix = "googletest-d175c8bf823e709d570772b038757fadf63bc632",
    build_file = "//third_party:googletest.BUILD",
)

new_http_archive(
    name = "lzma",
    urls = [
        "https://tukaani.org/xz/xz-5.2.3.tar.gz",
        "https://fossies.org/linux/misc/xz-5.2.3.tar.gz",
        "https://drive.google.com/uc?export=download&id=0Bxgu9m44g9qRNWVJcEZhUDJZVlU",
    ],
    sha256 = "71928b357d0a09a12a4b4c5fafca8c31c19b0e7d3b8ebb19622e96f26dbf28cb",
    strip_prefix = "xz-5.2.3/src",
    build_file = "//third_party:lzma.BUILD",
)

new_http_archive(
    name = "snappy",
    urls = ["https://github.com/google/snappy/archive/1.1.7.tar.gz"],
    sha256 = "3dfa02e873ff51a11ee02b9ca391807f0c8ea0529a4924afa645fbf97163f9d4",
    strip_prefix = "snappy-1.1.7",
    build_file = "//third_party:snappy.BUILD",
)

new_http_archive(
    name = "zlib",
    urls = ["https://www.zlib.net/zlib-1.2.11.tar.gz"],
    sha256 = "c3e5e9fdd5004dcb542feda5ee4f0ff0744628baf8ed2dd5d66f8ca1197cb1a1",
    strip_prefix = "zlib-1.2.11",
    build_file = "//third_party:zlib.BUILD",
)

new_http_archive(
    name = "boost",
    urls = ["https://dl.bintray.com/boostorg/release/1.65.1/source/boost_1_65_1.tar.gz"],
    sha256 = "a13de2c8fbad635e6ba9c8f8714a0e6b4264b60a29b964b940a22554705b6b60",
    strip_prefix = "boost_1_65_1",
    build_file = "//third_party:boost.BUILD",
)

mpi_repository(name = "mpi")
