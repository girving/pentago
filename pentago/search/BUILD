package(default_visibility = ["//visibility:public"])
load("//pentago:pentago.bzl", "cc_tests")

cc_library(
    name = "search",
    srcs = glob(["*.h", "*.cc"], exclude=["*_test.cc"]),
    copts = ["-std=c++20", "-Wall", "-Werror", "-fPIC", "-fno-stack-check"],
    deps = [
        "//pentago/base",
        "//pentago/data",
        "//pentago/utility",
    ],
)

cc_tests(
    names = [
        "supertable_test",
    ],
    deps = [
        ":search",
    ],
    size = "small",
)
