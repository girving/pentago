package(default_visibility = ["//visibility:public"])
load("//pentago:pentago.bzl", "cc_tests")

cc_binary(
    name = "generate",
    srcs = ["generate.cc"],
    copts = ["-std=c++20", "-Wall", "-Werror", "-fno-stack-check"],
    deps = [
        "//pentago/base",
        "//pentago/utility",
    ],
)

genrule(
    name = "public",
    outs = [
        "public/counts.svg",
        "public/favicon.svg",
        "public/index.html",
    ],
    srcs = [
        "index.html",
    ],
    tools = ["generate"],
    cmd = "$(location :generate) $(OUTS) $(SRCS)",
)
