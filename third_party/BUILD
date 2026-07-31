licenses(["notice"])  # BSD

exports_files(["googletest_char8t.patch", "boringssl_macos_arm_crypto.patch", "curl_warnings.patch"])

# Freestanding C++ headers for the wasm build in web/client
filegroup(
    name = "freestanding",
    srcs = glob(["freestanding/*"]),
    visibility = ["//visibility:public"],
)
