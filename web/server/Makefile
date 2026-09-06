# The server is plain Node.js except for xz.wasm, the WebAssembly xz decoder that bazel builds
# from liblzma (see BUILD.bazel).  Tests and ./deploy both need it.

BAZEL ?= bazel

all: test

.PHONY: all test test-all FORCE

# Always ask bazel, which knows the real dependencies (liblzma's sources included) and is
# quick when nothing has changed
xz.wasm: FORCE
	$(BAZEL) build //web/server:xz_wasm
	cp -f "$$($(BAZEL) info bazel-bin)/web/server/xz.wasm" $@
	chmod u+w,-x $@

# Offline tests
test: xz.wasm
	node unit.js

# All tests, including lookups against the Google Cloud Storage database
test-all: xz.wasm
	node unit.js all

FORCE:
