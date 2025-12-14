package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "explore",
    srcs = ["explore.cc"],
    copts = ["-std=c++20"],
    deps = [
        "//pentago/base",
        "//pentago/data",
        "//pentago/utility",
        "@cudd//:cplusplus",
    ],
)

cc_binary(
    name = "expand",
    srcs = ["expand.cc"],
    copts = ["-std=c++20"],
    deps = [
        "//pentago/base",
        "//pentago/data",
        "//pentago/utility",
    ],
)