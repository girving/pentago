package(default_visibility = ["//visibility:public"])
load("//pentago:pentago.bzl", "cc_tests")

cc_binary(
    name = "svgs",
    srcs = ["svgs.cc"],
    copts = ["-std=c++20", "-Wall", "-Werror", "-fno-stack-check"],
    deps = [
        "//pentago/base",
        "//pentago/utility",
    ],
)
