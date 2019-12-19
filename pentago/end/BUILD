package(default_visibility = ["//visibility:public"])
load("//pentago:pentago.bzl", "cc_tests")

cc_library(
    name = "end",
    srcs = glob(["*.h", "*.cc"], exclude=["check*.*", "options.*", "meaningless.cc", "make-indices.cc", "*_test.cc"]),
    copts = ["-std=c++1z", "-Wall", "-Werror", "-fPIC", "-fno-stack-check"],
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
    copts = ["-std=c++1z", "-Wall", "-Werror", "-fPIC", "-fno-stack-check"],
    deps = [],
)

cc_library(
    name = "check_lib",
    srcs = [
        "check.h",
        "check.cc",
    ],
    copts = ["-std=c++1z", "-Wall", "-Werror", "-fPIC", "-fno-stack-check"],
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
    copts = ["-std=c++1z", "-Wall", "-Werror", "-fno-stack-check"],
    deps = [
        ":end",
    ],
)

cc_library(
    name = "options",
    srcs = ["options.cc"],
    hdrs = ["options.h"],
    copts = ["-std=c++1z", "-Wall", "-Werror", "-fPIC", "-fno-stack-check"],
    deps = [
        ":config",
        "//pentago/base",
        "//pentago/utility",
    ],
)

cc_binary(
    name = "check",
    srcs = ["check_main.cc"],
    copts = ["-std=c++1z", "-Wall", "-Werror", "-fno-stack-check"],
    deps = [
        ":check_lib",
        ":end",
        ":options",
        "//pentago/high",
    ],
)

cc_binary(
    name = "make-indices",
    srcs = ["make-indices.cc"],
    copts = ["-std=c++1z", "-Wall", "-Werror", "-fno-stack-check"],
    deps = [
        "//pentago/data",
        "//pentago/high",
    ],
)
