package(default_visibility = ["//visibility:public"])
load("//pentago:pentago.bzl", "cc_tests")

cc_library(
    name = "high",
    srcs = glob(["*.h", "*.cc"], exclude=["check.*", "*_test.cc"]),
    copts = ["-std=c++1z", "-Wall", "-Werror", "-fPIC", "-fno-stack-check"],
    deps = [
        "//pentago/base",
        "//pentago/data",
        "//pentago/end",
        "//pentago/mid:halfsuper",
        "//pentago/utility",
    ],
)

cc_library(
    name = "check",
    srcs = glob(["check.h", "check.cc"]),
    copts = ["-std=c++1z", "-Wall", "-Werror", "-fPIC", "-fno-stack-check"],
    deps = [
        ":high",
        "//pentago/base",
        "//pentago/data",
        "//pentago/search",
        "//pentago/utility",
    ],
)

cc_tests(
    names = [
        "high_test",
    ],
    deps = [
        ":high",
        ":check",
    ],
    size = "small",
)
