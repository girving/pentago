package(default_visibility = ["//visibility:public"])
load("//pentago:pentago.bzl", "cc_tests")

cc_binary(
    name = "precompute",
    srcs = ["precompute.cc"],
    copts = ["-std=c++1z", "-Wall", "-Werror", "-fPIC", "-fno-stack-check"],
    deps = [
        "//pentago/utility",
    ],
)

genrule(
    name = "tables",
    outs = [
        "gen/tables.h",
        "gen/tables.cc",
    ],
    tools = [":precompute"],
    cmd = "$(location :precompute) $(OUTS)",
)

cc_library(
    name = "base",
    srcs = glob(["*.h", "*.cc"], exclude=["precompute.cc", "*_test.cc"]) + ["gen/tables.h", "gen/tables.cc"],
    copts = ["-std=c++1z", "-Wall", "-Werror", "-fPIC", "-fno-stack-check"],
    deps = [
        "//pentago/utility",
    ],
)

cc_tests(
    names = [
        "all_boards_test",
        "pentago_test",
        "super_test",
    ],
    deps = [
        ":base",
        "//pentago/utility",
    ],
)
