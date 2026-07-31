load("@rules_cc//cc:defs.bzl", "cc_binary")
load("//pentago:pentago.bzl", "cc_tests")
package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "svgs",
    srcs = ["svgs.cc", "tags.h"],
    copts = ["-std=c++20", "-Wall", "-Werror", "-fno-stack-check"],
    deps = [
        "//pentago/base",
        "//pentago/utility",
    ],
)

cc_binary(
    name = "site",
    srcs = ["site.cc", "tags.h"],
    copts = ["-std=c++20", "-Wall", "-Werror", "-fno-stack-check"],
    deps = [
        "//pentago/utility",
    ],
)
