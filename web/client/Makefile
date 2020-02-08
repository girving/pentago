all: public test

public: public/client.js public/board.css public/main.css

public/client.js: client.js *.js mid.wasm
	node_modules/.bin/browserify -t brfs $< -o $@

mid.wasm: build-wasm
	./build-wasm

public/%.css: %.styl
	node_modules/.bin/stylus --compress --use nib < $< > $@

.PHONY: test clean

test: mid.wasm
	node unit.js

clean:
	rm -f public/client.js public/mid_worker.js public/board.css public/main.css mid.wasm
