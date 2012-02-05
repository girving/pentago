all: pentago

CXX = g++
CXXFLAGS = -Wall -Werror -O3

pentago: pentago.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

.PHONY: clean

clean:
	rm -f pentago *.o *.E *.so
