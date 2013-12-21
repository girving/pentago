all: bundle.js test

bundle.js: client.js board.js
	browserify $< -o $@

.PHONY: test clean

test:
	node unit.js

clean:
	rm -f bundle.js
