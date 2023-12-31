all: public/mid.wasm build/tests.wasm public/counts.svg

public/mid.wasm build/tests.wasm: build-wasm
	./build-wasm

public/counts.svg: cc/svgs.cc
	bazel run cc/svgs counts > $@
