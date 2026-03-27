load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")
load("//pentago:pentago.bzl", "cc_tests", "COPTS")
package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "precompute",
    srcs = ["precompute.cc"],
    copts = COPTS,
    deps = [
        "//pentago/utility",
    ],
)

genrule(
    name = "tables",
    outs = [
        "gen/tables.h",
        "gen/tables.cc",
        "gen/tables.js",
        "gen/halfsuper_wins.h",
    ],
    tools = [":precompute"],
    cmd = "$(location :precompute) $(OUTS)",
)

cc_library(
    name = "halfsuper_wins",
    hdrs = ["gen/halfsuper_wins.h"],
)

cc_library(
    name = "base",
    srcs = glob(["*.h", "*.cc"], exclude=["precompute.cc", "*_test.cc"]) + ["gen/tables.h", "gen/tables.cc"],
    copts = COPTS,
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
