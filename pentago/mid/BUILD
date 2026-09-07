load("@rules_cc//cc:defs.bzl", "cc_library")
load("//pentago:pentago.bzl", "cc_tests", "COPTS")
package(default_visibility = ["//visibility:public"])

cc_library(
    name = "mid",
    srcs = ["internal.h", "internal_c.h", "midengine.h", "subsets.h", "subsets_c.h", "midengine.cc"],
    copts = COPTS,
    deps = [
        ":halfsuper",
        "//pentago/base",
        "//pentago/high",
        "//pentago/utility",
    ],
)

cc_library(
    name = "codec",
    srcs = ["codec.h", "codec.cc"],
    copts = COPTS,
    deps = [":halfsuper", "//pentago/utility"],
)

cc_library(
    name = "tiled",
    srcs = ["tiled.h", "tiled.cc"],
    copts = COPTS,
    deps = [":codec", ":halfsuper", ":mid", "//pentago/base", "//pentago/high", "//pentago/utility"],
)

cc_library(
    name = "halfsuper",
    srcs = ["halfsuper.h", "halfsuper_c.h", "halfsuper.cc"],
    copts = COPTS,
    deps = [
        "//pentago/base",
        "//pentago/base:halfsuper_wins",
        "//pentago/data",
        "//pentago/utility",
    ],
)

cc_tests(
    names = [
        "mid_test",
        "benchmark_test",
    ],
    deps = [
        ":codec",
        ":halfsuper",
        ":mid",
        ":tiled",
        "//pentago/high:check",
    ],
)

cc_tests(
    names = ["tiled_test"],
    # Solves several boards both ways, which needs more than the medium budget on slow CI runners
    size = "large",
    deps = [
        ":codec",
        ":halfsuper",
        ":mid",
        ":tiled",
        "//pentago/high:check",
    ],
)

# All sources and headers, for the freestanding wasm build in web/client
filegroup(
    name = "srcs",
    srcs = glob(["*.h", "*.cc"]),
)
