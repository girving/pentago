package(default_visibility = ["//visibility:public"])
load("//pentago:pentago.bzl", "cc_tests", "COPTS")

cc_library(
    name = "mpi",
    srcs = glob(["*.h", "*.cc"], exclude=["main.cc", "*_test.cc"]),
    copts = COPTS + ["-Wno-deprecated-volatile"],
    deps = [
        "//pentago/base",
        "//pentago/end",
        "//pentago/end:check_lib",
        "//pentago/end:options",
        "//pentago/search",
        "//pentago/utility",
        "@mpi//:mpi",
    ],
)

cc_binary(
    name = "endgame-mpi",
    srcs = ["main.cc"],
    copts = COPTS,
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
        "//pentago/base",
        "//pentago/utility",
    ],
)
