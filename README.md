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

2. We consider two types of simplified games:

   a. Boring defense: White always chooses to undo black's last rotation,
      and 5-in-a-rows by white are ignored.  Under these rules, rotations
      can be ignored when generating moves, except that we have allow black
      one rotation when checking for a win.

   b. Boring offense: Black always chooses to under white's last rotation,
      and 5-in-a-rows are considered on both sides.  However, black 5-in-a-rows
      are ignored if they occur as a consequence of a white rotation.  As before,
      this allows rotations to be ignored, except that we have to allow white
      (but not black) one rotation when checking for a win.

   Starting from any position, a tie for white under "boring defense" rules
   guarantees a tie for white under normal rules.  Similarly, a win for black
   under "boring offense" rules guarantees a win for black under normal rules.
   Therefore, we can use either set of rules to prune away hopefully large
   branches of the tree, taking advantage of the reduced branching factor
   of the simplified games.

   I had originally hoped that boring defense would succeed starting from an
   empty board, but that turns out not to be the case: if white plays boring
   defense, black wins in 15 ply.
