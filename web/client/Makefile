all: public/mid.wasm build/tests.wasm bazel

public/mid.wasm build/tests.wasm: build-wasm
	./build-wasm

.PHONY: bazel
bazel:
	cd cc && bazel build ...
