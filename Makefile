all: engine.so

CXX = g++
CXXFLAGS = -Wall -Werror -O3

engine.so: engine.cpp gen/win.h gen/rotate.h gen/pack.h gen/unpack.h gen/move.h
	python setup.py build
	cp build/lib.*/engine.so .

gen/%.h: precompute
	./precompute $@

.PHONY: clean

clean:
	rm -f pentago *.o *.E *.so
