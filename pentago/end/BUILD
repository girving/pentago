package(default_visibility = ["//visibility:public"])
load("//pentago:pentago.bzl", "cc_tests", "COPTS")

cc_library(
    name = "end",
    srcs = glob(["*.h", "*.cc"], exclude=["check*.*", "options.*", "meaningless.cc", "make-indices.cc", "*_test.cc"]),
    copts = COPTS,
    deps = [
        "//pentago/base",
        "//pentago/data",
        "//pentago/search",
        "//pentago/utility",
        "@snappy//:snappy",
    ],
)

cc_library(
    name = "config",
    srcs = ["config.h"],
    copts = COPTS,
    deps = [],
)

cc_library(
    name = "check_lib",
    srcs = [
        "check.h",
        "check.cc",
    ],
    copts = COPTS,
    deps = [
        ":end",
    ],
)

cc_tests(
    names = [
        "end_test",
    ],
    deps = [
        ":check_lib",
        ":end",
        "//pentago/base",
        "//pentago/utility",
    ],
)

cc_binary(
    name = "meaningless",
    srcs = ["meaningless.cc"],
    copts = COPTS,
    deps = [
        ":end",
    ],
)

cc_library(
    name = "options",
    srcs = ["options.cc"],
    hdrs = ["options.h"],
    copts = COPTS,
    deps = [
        ":config",
        "//pentago/base",
        "//pentago/utility",
    ],
)

cc_binary(
    name = "check",
    srcs = ["check_main.cc"],
    copts = COPTS,
    deps = [
        ":check_lib",
        ":end",
        ":options",
        "//pentago/high",
        "//pentago/high:check",
    ],
)

cc_binary(
    name = "make-indices",
    srcs = ["make-indices.cc"],
    copts = COPTS,
    deps = [
        "//pentago/data",
        "//pentago/high",
    ],
)
