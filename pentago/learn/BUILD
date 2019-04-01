package(default_visibility = ["//visibility:public"])
load("//pentago:pentago.bzl", "cc_tests")

cc_binary(
    name = "libpentago_ops.so",
    srcs = ["ops.cc"],
    copts = ["-std=c++1z", "-Wall", "-Werror", "-fPIC"],
    deps = [
        "//pentago/base",
        "//pentago/end",
        "//pentago/high",
        "//pentago/utility",
        "@tensorflow//:tensorflow",
    ],
    linkshared=True,
)
