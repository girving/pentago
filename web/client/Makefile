all: public/mid.wasm build/tests.wasm public/counts.svg public/favicon.svg

public/mid.wasm build/tests.wasm: build-wasm
	./build-wasm

public/counts.svg: cc/svgs.cc
	bazel run cc/svgs counts > $@

public/favicon.svg: cc/svgs.cc
	bazel run cc/svgs favicon > $@
