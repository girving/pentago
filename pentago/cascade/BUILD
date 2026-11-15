load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")
load("//pentago:pentago.bzl", "cc_tests", "COPTS")
package(default_visibility = ["//visibility:public"])

cc_library(
    name = "tie",
    srcs = ["tie.h", "tie.cc"],
    copts = COPTS,
    deps = [
        "//pentago/base",
        "//pentago/high",
        "//pentago/utility",
    ],
)

cc_library(
    name = "cascade",
    srcs = ["cascade.h", "cascade.cc"],
    copts = COPTS,
    deps = [
        ":tie",
        "//pentago/base",
        "//pentago/high",
        "//pentago/mid",
        "//pentago/mid:halfsuper",
        "//pentago/search",
        "//pentago/utility",
    ],
)

cc_tests(
    names = [
        "tie_test",
    ],
    deps = [
        ":tie",
        "//pentago/utility",
    ],
    size = "small",
)

cc_tests(
    names = [
        "cascade_test",
    ],
    deps = [
        ":cascade",
        "//pentago/utility",
    ],
    size = "small",
)

# Generators for the stored test positions
cc_binary(
    name = "gen_tie_cases",
    srcs = ["gen_tie_cases.cc"],
    copts = COPTS,
    deps = [":tie", "//pentago/base", "//pentago/mid", "//pentago/utility"],
)

cc_binary(
    name = "gen_cascade_cases",
    srcs = ["gen_cascade_cases.cc"],
    copts = COPTS,
    deps = [":cascade", "//pentago/base", "//pentago/utility"],
)
