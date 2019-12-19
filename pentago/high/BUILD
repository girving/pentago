package(default_visibility = ["//visibility:public"])
load("//pentago:pentago.bzl", "cc_tests")

cc_library(
    name = "high",
    srcs = glob(["*.h", "*.cc"], exclude=["*_test.cc"]),
    copts = ["-std=c++1z", "-Wall", "-Werror", "-fPIC", "-fno-stack-check"],
    deps = [
        "//pentago/base",
        "//pentago/data",
        "//pentago/end",
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
    ],
    size = "small",
)
