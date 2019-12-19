package(default_visibility = ["//visibility:public"])
load("//pentago:pentago.bzl", "cc_tests")

cc_library(
    name = "mid",
    srcs = glob(["*.h", "*.cc"], exclude=["*_test.cc"]),
    copts = ["-std=c++1z", "-Wall", "-Werror", "-fPIC", "-fno-stack-check"],
    deps = [
        "//pentago/base",
        "//pentago/data",
        "//pentago/high",
        "//pentago/search",
        "//pentago/utility",
    ],
)

cc_tests(
    names = [
        "mid_test",
        "benchmark_test",
    ],
    deps = [
        ":mid",
    ],
)
