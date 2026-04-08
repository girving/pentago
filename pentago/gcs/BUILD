load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")
load("//pentago:pentago.bzl", "COPTS")
package(default_visibility = ["//visibility:public"])

cc_library(
    name = "gcs",
    srcs = ["gcs.cc"],
    hdrs = ["gcs.h", "internal.h"],
    copts = COPTS,
    deps = [
        "//pentago/data",
        "//pentago/utility",
        "@curl",
        "@boringssl//:crypto",
    ],
)

cc_binary(
    name = "gcs_sharder_test",
    srcs = ["gcs_sharder_test.cc"],
    copts = COPTS,
    deps = [
        ":gcs",
        "//pentago/data",
        "//pentago/shard",
        "//pentago/utility",
    ],
    data = [
        "//pentago/shard:sharder",
        "//data:slices",
    ],
)

cc_test(
    name = "gcs_test",
    srcs = ["gcs_test.cc"],
    copts = COPTS + ["-Wsign-compare"],
    linkstatic = True,
    deps = [
        ":gcs",
        "//pentago/utility",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
    size = "small",
)
