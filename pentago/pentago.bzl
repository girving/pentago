# Pentago build extensions

# Common copts for all pentago targets
COPTS = ["-std=c++20", "-Wall", "-Werror", "-fPIC", "-fno-stack-check"] + select({
    "@bazel_tools//tools/cpp:clang": ["-Wno-vla-cxx-extension"],
    "//conditions:default": [],
})

def cc_tests(names, deps, data=[], size="medium"):
  deps = deps + ["@com_google_googletest//:gtest", "@com_google_googletest//:gtest_main"]
  copts = COPTS + ["-Wsign-compare"]
  linkopts = select({
      "@bazel_tools//tools/cpp:clang": ["-Wno-unused-command-line-argument"],
      "//conditions:default": [],
  })
  for name in names:
    native.cc_test(name=name, srcs=[name + ".cc"], copts=copts, linkopts=linkopts, deps=deps,
                   data=data, size=size)
