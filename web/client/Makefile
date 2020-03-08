all: public/main.css public/mid.wasm

public/main.css: src/main.styl
	node_modules/.bin/stylus --compress --use nib < $< > $@

public/mid.wasm: build-wasm
	./build-wasm
