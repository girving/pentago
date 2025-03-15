package(default_visibility = ["//visibility:public"])
load("//pentago:pentago.bzl", "cc_tests")

cc_library(
    name = "data",
    srcs = glob(["*.h", "*.cc"], exclude=["roundtrip.cc", "*_test.cc"]),
    copts = ["-std=c++20", "-Wall", "-Werror", "-fPIC", "-fno-stack-check"],
    deps = [
        "//pentago/base",
        "//pentago/utility",
        "@lzma//:lzma",
        "@zlib//:zlib",
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
    copts = ["-std=c++20", "-Wall", "-Werror", "-fno-stack-check"],
    deps = [
        ":data",
        "//pentago/end:options",
    ],
)
