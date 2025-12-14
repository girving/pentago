licenses(["notice"])  # BSD

# CUDD - Colorado University Decision Diagram package

# Generate config.h with necessary defines
genrule(
    name = "config_h",
    outs = ["config.h"],
    cmd = """
cat > $@ << 'EOF'
/* Generated config.h for CUDD */
#ifndef CUDD_CONFIG_H
#define CUDD_CONFIG_H

/* Standard headers we need */
#include <stddef.h>
#include <stdint.h>
#include <assert.h>

#define PACKAGE_VERSION "3.0.0"
#define HAVE_IEEE_754 1
#define HAVE_POWL 1  /* Use system powl from math.h */
#define SIZEOF_VOID_P 8
#define SIZEOF_INT 4
#define SIZEOF_LONG 8
#define HAVE_ASSERT_H 1
#define HAVE_STDINT_H 1
#define HAVE_INTTYPES_H 1
#define HAVE_STDLIB_H 1
#define HAVE_STRING_H 1
#define HAVE_SYS_RESOURCE_H 1
#define HAVE_SYS_STAT_H 1
#define HAVE_UNISTD_H 1

#endif /* CUDD_CONFIG_H */
EOF
""",
)

# Common copts for all CUDD libraries
CUDD_COPTS = [
    "-w",  # Suppress warnings in third-party code
    "-I$(GENDIR)/external/cudd",
]

cc_library(
    name = "util",
    srcs = glob(["util/*.c"], exclude = ["util/test*.c"]) + [":config_h"],
    hdrs = glob(["util/*.h"]),
    includes = [".", "util"],
    copts = CUDD_COPTS,
)

cc_library(
    name = "st",
    srcs = glob(["st/*.c"], exclude = ["st/test*.c"]),
    hdrs = glob(["st/*.h"]),
    includes = ["st"],
    deps = [":util"],
    copts = CUDD_COPTS,
)

cc_library(
    name = "epd",
    srcs = glob(["epd/*.c"], exclude = ["epd/test*.c"]),
    hdrs = glob(["epd/*.h"]),
    includes = ["epd"],
    deps = [":util"],
    copts = CUDD_COPTS,
)

cc_library(
    name = "mtr",
    srcs = glob(["mtr/*.c"], exclude = ["mtr/test*.c"]),
    hdrs = glob(["mtr/*.h"]),
    includes = ["mtr"],
    deps = [":util"],
    copts = CUDD_COPTS,
)

cc_library(
    name = "cudd",
    srcs = glob(["cudd/*.c"], exclude = ["cudd/test*.c"]),
    hdrs = glob(["cudd/*.h"]),
    includes = ["cudd"],
    deps = [":util", ":st", ":epd", ":mtr"],
    copts = CUDD_COPTS,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cplusplus",
    srcs = ["cplusplus/cuddObj.cc"],
    hdrs = ["cplusplus/cuddObj.hh"],
    includes = ["cplusplus"],
    deps = [":cudd"],
    copts = CUDD_COPTS,
    visibility = ["//visibility:public"],
)
