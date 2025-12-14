# Pentago build extensions

# Common copts for all pentago targets
COPTS = ["-std=c++20", "-Wall", "-Werror", "-Wno-vla-cxx-extension", "-fPIC", "-fno-stack-check"]

def cc_tests(names, deps, data=[], size="medium"):
  deps = deps + ["@com_google_googletest//:gtest", "@com_google_googletest//:gtest_main"]
  copts = COPTS + ["-Wsign-compare"]
  linkopts = ["-Wno-unused-command-line-argument"]
  for name in names:
    native.cc_test(name=name, srcs=[name + ".cc"], copts=copts, linkopts=linkopts, deps=deps,
                   data=data, size=size)
