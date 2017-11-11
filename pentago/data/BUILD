package(default_visibility = ["//visibility:public"])
load("//pentago:pentago.bzl", "cc_tests")

cc_library(
    name = "data",
    srcs = glob(["*.h", "*.cc"], exclude=["roundtrip.cc", "async_block_cache.cc", "*_test.cc"]),
    copts = ["-std=c++1z", "-Wall", "-Werror", "-fPIC"],
    deps = [
        "//pentago/base",
        "//pentago/utility",
        "@lzma//:lzma",
        "@zlib//:zlib",
    ],
)

cc_library(
    name = "async",
    srcs = ["async_block_cache.cc"],
    copts = ["-std=c++1z", "-Wall", "-Werror", "-fPIC"],
    deps = [
        ":data",
        "//pentago/end:config",
        "//pentago/high",
    ],
)

cc_tests(
    names = [
        "compress_test",
        "filter_test",
        "lru_test",
        "numpy_test",
        "supertensor_test",
    ],
    deps = [
        ":data",
        "//pentago/base",
        "//pentago/utility",
    ],
    size = "small",
)

cc_binary(
    name = "roundtrip",
    srcs = ["roundtrip.cc"],
    copts = ["-std=c++1z", "-Wall", "-Werror"],
    deps = [
        ":data",
        "//pentago/mpi:options",
    ],
)
