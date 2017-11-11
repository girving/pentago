# Pentago utility code

package(default_visibility = ["//visibility:public"])
load("//pentago:pentago.bzl", "cc_tests")

cc_library(
    name = "utility",
    srcs = glob(["*.h", "*.cc"], exclude=["*_test.cc"]),
    copts = ["-std=c++1z", "-Wall", "-Werror", "-fPIC"],
    linkopts = ["-lpthread"],
    deps = [
        "@random123//:random123",
        "@tinyformat//:tinyformat",
        "@boost//:boost",
    ],
)

cc_tests(
    names = [
        "aligned_test",
        "thread_test",
    ],
    deps = [
        ":utility",
    ],
    size = "small",
)
