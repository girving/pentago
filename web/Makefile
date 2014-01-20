all: bundle.js test board.css main.css

bundle.js: client.js board.js
	browserify $< -o $@

%.css: %.styl
	node_modules/.bin/stylus --compress --use nib < $< > $@

.PHONY: test clean

test:
	node unit.js

clean:
	rm -f bundle.js board.css main.css
