all: test public

public: public/client.js public/mid_worker.js public/board.css public/main.css

public/client.js: client.js *.js
	node_modules/.bin/browserify $< -o $@

public/mid_worker.js: mid_worker.js *.js
	node_modules/.bin/browserify $< -o $@

public/%.css: %.styl
	node_modules/.bin/stylus --compress --use nib < $< > $@

.PHONY: test clean

test:
	node unit.js

clean:
	rm -f public/client.js public/mid_worker.js public/board.css public/main.css
