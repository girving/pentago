package(default_visibility = ["//visibility:public"])
load("//pentago:pentago.bzl", "cc_tests")

cc_library(
    name = "mpi",
    srcs = glob(["*.h", "*.cc"], exclude=["main.cc", "options.*", "trace.*", "*_test.cc"]),
    copts = ["-std=c++1z", "-Wall", "-Werror"],
    deps = [
        ":options",
        "//pentago/base",
        "//pentago/end",
        "//pentago/end:check_lib",
        "//pentago/search",
        "//pentago/utility",
        "//third_party/mpi",
    ],
)

cc_library(
    name = "trace",
    srcs = [
        "trace.h",
        "trace.cc"
    ],
    copts = ["-std=c++1z", "-Wall", "-Werror"],
    deps = [
        "//pentago/end:config",
        "//pentago/utility",
    ],
)

cc_library(
    name = "options",
    srcs = ["options.cc"],
    hdrs = ["options.h"],
    copts = ["-std=c++1z", "-Wall", "-Werror"],
    deps = [
        "//pentago/base",
        "//pentago/end:config",
        "//pentago/utility",
    ],
)

cc_binary(
    name = "endgame-mpi",
    srcs = ["main.cc"],
    copts = ["-std=c++1z", "-Wall", "-Werror"],
    deps = [
        ":mpi",
    ],
)

cc_tests(
    names = [
        "mpi_test",
    ],
    data = [
        ":endgame-mpi",
        "//pentago/end:check",
    ],
    deps = [
        ":mpi",
        "//pentago/base",
        "//pentago/utility",
    ],
)
