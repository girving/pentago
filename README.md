A simple pentago player
=======================

A simple pentago player using (essentially) alpha-beta search, a transposition
table, and various precomputed lookup tables for move evaluating, win detection,
etc.  For now it's extremely weak.

For the rules of pentago, see http://en.wikipedia.org/wiki/Pentago.

### Dependencies

The core engine is written in C++ and exposed to Python as an extension module.
The rest of the code (tests, interface, etc.) is in Python.  The single direct
dependency is on the Otherlab core libraries (`other`).  Unfortunately, while the
used portion of `other may be open source at some point, it is not publically
available at this time.

### Usage

To configure, simply symlink the build system over from `other`:

    cd pentago
    ln -s $OTHER/SConstruct
    ln -s $OTHER/SConstruct.options # optional

To build, run

    scons

To run the unit tests, instead [py.test](http://pytest.org) and run

    py.test

To run a fully automated game (computer vs. itself), run

    ./pentago

More to follow.

### Algorithm notes

1. Pentago has an inconvenient number of spaces, namely 36 instead
   of 32.  We could potentially dodge this problem by templatizing
   over the value of the center space in each quadrant.  This is
   almost surely a win, but I'll stick to 2 64-bit integers for now
   and revisit that trick later.  It would be easy if I knew that
   the first four optimal moves were center moves, but I don't have
   a proof of that.
