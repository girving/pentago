load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")
load("//pentago:pentago.bzl", "cc_tests", "COPTS")
package(default_visibility = ["//visibility:public"])

cc_library(
    name = "data",
    srcs = glob(["*.h", "*.cc"], exclude=["roundtrip.cc", "sharder.cc", "shard_sizes.cc", "*_test.cc"]),
    copts = COPTS,
    deps = [
        "//pentago/base",
        "//pentago/utility",
        "@lzma//:lzma",
        "@zlib//:zlib",
    ],
)

cc_tests(
    names = [
        "arithmetic_test",
        "compress_test",
        "filter_test",
        "lru_test",
        "numpy_test",
        "shard_test",
        "supertensor_test",
        "ternary_test",
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
    copts = COPTS,
    deps = [
        ":data",
        "//pentago/end:options",
    ],
)

cc_binary(
    name = "shard_sizes",
    srcs = ["shard_sizes.cc"],
    copts = COPTS,
    deps = [
        ":data",
        "//pentago/base",
        "//pentago/utility",
    ],
)

cc_binary(
    name = "sharder",
    srcs = ["sharder.cc"],
    copts = COPTS,
    deps = [
        ":data",
        "//pentago/base",
        "//pentago/utility",
    ],
)

cc_test(
    name = "sharder_test",
    srcs = ["sharder_test.cc"],
    copts = COPTS + ["-Wsign-compare"],
    deps = [
        ":data",
        "//pentago/base",
        "//pentago/utility",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
    data = [
        ":sharder",
        "//data:slices",
    ],
    size = "medium",
)
