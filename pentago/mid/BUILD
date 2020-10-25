package(default_visibility = ["//visibility:public"])
load("//pentago:pentago.bzl", "cc_tests")

cc_library(
    name = "mid",
    srcs = glob(["*.h", "*.cc"], exclude=["*_test.cc", "halfsuper*.*"]),
    copts = ["-std=c++1z", "-Wall", "-Werror", "-fPIC", "-fno-stack-check"],
    deps = [
        ":halfsuper",
        "//pentago/base",
        "//pentago/high",
        "//pentago/search",
        "//pentago/utility",
    ],
)

cc_library(
    name = "halfsuper",
    srcs = ["halfsuper.h", "halfsuper_c.h", "halfsuper.cc"],
    copts = ["-std=c++1z", "-Wall", "-Werror", "-fPIC", "-fno-stack-check"],
    deps = [
        "//pentago/base",
        "//pentago/base:halfsuper_wins",
        "//pentago/data",
        "//pentago/utility",
    ],
)

cc_tests(
    names = [
        "mid_test",
        "benchmark_test",
    ],
    deps = [
        ":halfsuper",
        ":mid",
    ],
)
