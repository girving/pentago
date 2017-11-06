package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # BSD-3

genrule(
    name = "stubs",
    srcs = ["snappy-stubs-public.h.in"],
    outs = ["snappy-stubs-public.h"],
    cmd = "sed -e 's/\$${HAVE_.*}/1/g' -e 's/\$${SNAPPY_MAJOR}/1/' " +
          "-e 's/\$${SNAPPY_MINOR}/1/' -e 's/\$${SNAPPY_PATCHLEVEL}/7/' $< > $@",
)

cc_library(
    name = "snappy",
    srcs = [
        "snappy-stubs-public.h",
        "snappy-c.cc",
        "snappy-internal.h",
        "snappy-sinksource.cc",
        "snappy-stubs-internal.cc",
        "snappy-stubs-internal.h",
        "snappy.cc",
        "snappy-sinksource.h",
        "snappy-c.h", 
    ],
    hdrs = ["snappy.h"],
    copts = [],
    includes = ["api"],
)
