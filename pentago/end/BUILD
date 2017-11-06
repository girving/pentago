package(default_visibility = ["//visibility:public"])
load("//pentago:pentago.bzl", "cc_tests")

cc_library(
    name = "end",
    srcs = glob(["*.h", "*.cc"], exclude=["check*.*", "meaningless.cc", "*_test.cc"]),
    copts = ["-std=c++1z", "-Wall", "-Werror"],
    deps = [
        "//pentago/base",
        "//pentago/data",
        "//pentago/mpi:trace",
        "//pentago/search",
        "//pentago/utility",
        "@snappy//:snappy",
    ],
)

cc_library(
    name = "config",
    srcs = ["config.h"],
    copts = ["-std=c++1z", "-Wall", "-Werror"],
    deps = [],
)

cc_library(
    name = "check_lib",
    srcs = [
        "check.h",
        "check.cc",
    ],
    copts = ["-std=c++1z", "-Wall", "-Werror"],
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
    copts = ["-std=c++1z", "-Wall", "-Werror"],
    deps = [
        ":end",
    ],
)

cc_binary(
    name = "check",
    srcs = ["check_main.cc"],
    copts = ["-std=c++1z", "-Wall", "-Werror"],
    deps = [
        ":check_lib",
        ":end",
        "//pentago/high",
        "//pentago/mpi:options",
    ],
)
