# Pentago utility code

package(default_visibility = ["//visibility:public"])
load("//pentago:pentago.bzl", "cc_tests", "COPTS")

cc_library(
    name = "utility",
    srcs = glob(["*.h", "*.cc"], exclude=["*_test.cc"]),
    copts = COPTS + ["-Wno-deprecated-volatile"],
    linkopts = ["-lpthread"],
    deps = [
        "@random123//:random123",
        "@tinyformat//:tinyformat",
    ],
)

cc_tests(
    names = [
        "aligned_test",
        "random_test",
        "thread_test",
    ],
    deps = [
        ":utility",
    ],
    size = "small",
)
