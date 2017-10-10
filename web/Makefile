all: test public

public: public/bundle.js public/board.css public/main.css

public/bundle.js: client.js board.js
	browserify $< -o $@

public/%.css: %.styl
	node_modules/.bin/stylus --compress --use nib < $< > $@

.PHONY: test clean

test:
	node unit.js

clean:
	rm -f public/bundle.js public/board.css public/main.css
