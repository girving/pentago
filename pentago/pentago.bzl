# Pentago build extensions

def cc_tests(names, deps, data=[], size="medium"):
  deps = deps + ["@com_google_googletest//:gtest", "@com_google_googletest//:gtest_main"]
  copts = ["-std=c++1z", "-Werror"]
  linkopts = ["-Wno-unused-command-line-argument"]
  for name in names:
    native.cc_test(name=name, srcs=[name + ".cc"], copts=copts, linkopts=linkopts, deps=deps,
                   data=data, size=size)
