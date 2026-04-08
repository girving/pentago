load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")
load("//pentago:pentago.bzl", "cc_tests", "COPTS")
package(default_visibility = ["//visibility:public"])

cc_library(
    name = "shard",
    srcs = glob(["*.h", "*.cc"], exclude=["sharder.cc", "shard_sizes.cc", "shard_permute_search.cc", "*_test.cc", "*_bench.cc"]),
    copts = COPTS,
    deps = [
        "//pentago/base",
        "//pentago/data",
        "//pentago/utility",
    ],
)

cc_tests(
    names = [
        "arithmetic_test",
        "shard_test",
        "ternary_test",
    ],
    deps = [
        ":shard",
        "//pentago/base",
        "//pentago/utility",
    ],
    size = "small",
)

cc_test(
    name = "shard_permute_test",
    srcs = ["shard_permute_test.cc"],
    copts = COPTS + ["-Wsign-compare"],
    deps = [
        ":shard",
        "//pentago/base",
        "//pentago/utility",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
    size = "medium",
)

cc_test(
    name = "shard_bench",
    srcs = ["shard_bench.cc"],
    copts = COPTS + ["-Wsign-compare"],
    deps = [
        ":shard",
        "//pentago/base",
        "//pentago/utility",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
    size = "small",
)

cc_binary(
    name = "shard_sizes",
    srcs = ["shard_sizes.cc"],
    copts = COPTS,
    deps = [
        ":shard",
        "//pentago/base",
        "//pentago/data",
        "//pentago/utility",
    ],
    data = ["//data:counts"],
)

cc_binary(
    name = "shard_permute_search",
    srcs = ["shard_permute_search.cc"],
    copts = COPTS,
    deps = [":shard"],
)

cc_binary(
    name = "sharder",
    srcs = ["sharder.cc"],
    copts = COPTS,
    deps = [
        ":shard",
        "//pentago/base",
        "//pentago/data",
        "//pentago/utility",
    ],
)

cc_test(
    name = "sharder_test",
    srcs = ["sharder_test.cc"],
    copts = COPTS + ["-Wsign-compare"],
    deps = [
        ":shard",
        "//pentago/base",
        "//pentago/data",
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
