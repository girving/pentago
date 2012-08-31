A brute force pentago solver
============================

The goal of this project is a strong solution of the board game pentago, which
means a practical algorithm for perfect play starting from any position.
For the rules of pentago, see http://en.wikipedia.org/wiki/Pentago.

The code consists of two parts: a serial forward engine capable of search out to
depth 17, and a parallel out-of-core backward engine suitable for endgame
database computation on hundreds to thousands of machines.  The main features
of the forward engine are

1. Alpha-beta search specialized to binary choices (win vs. tie or tie vs. loss,
   equivalent to MTD(f) for the case of three possible outcomes).

2. Simultaneous evaluation of all 256 different rotated versions of a given
   board in parallel using SSE instructions.  This reduces the branching factor
   by a factor of 8, which is critical in both the forward and backward engines.

3. Symmetry-aware transposition tables, using the full 2048 element symmetry
   group of global+local transformations.

Starting from an empty board, the forward engine takes about 8.2 hours on a 2.3
GHz Macbook Pro to prove that pentago is a tie within the first 17 moves.  Since
forward search cost increases exponentially with depth, it would have no hope
of establishing the value of the game unless it ended within a few ply after
depth 17, even with large parallel resources.  Therefore, I switched to a
backward solver, attempting to enumerate the entire state space of 3e15
positions (after symmetry reduction).  The backwards endgame solver will be
described in more detail once it I try it out on a suitable supercomputer.

### Dependencies

The code is a mix of pure C++ (for MPI code), C++ routines exposed to Python, and
Python code for tests, interfaces, etc.  The direct dependencies are

* [other/core](https://github.com/otherlab/core): Otherlab core utilities and Python bindings
* [mpi](http://en.wikipedia.org/wiki/Message_Passing_Interface): OpenMPI, MPICH2, etc.
* [zlib](http://zlib.net): Okay but slow lossless compression.
* [xz](http://tukaani.org/xz): Good but slower lossless compression
* [snappy](http://code.google.com/p/snappy): Fast but poor lossless compression

other/core has a few additional indirect dependencies (python, numpy, boost).

### Setup

1. Install `other/core` and the other dependencies.
2. Configure `other/core` to be reference-count thread safe, by adding

        thread_safe = 1

   to `other/config.py`.  Rebuild `other/core` if necessary.

3. Setup and build `pentago`:

        cd pentago
        $OTHER/core/build/setup
        scons -j 5

4. Test.  Note that the mpi tests will fail if mpirun/aprun does not exist on the host machine.

        py.test

### Usage

The interface consists of a variety of python scripts.  I'll give a few
examples of their usage here; for more details run "script -h".

1. pentago: Main forward evaluation driver script

        ./pentago -d 11 # Play a low depth automated game against itself
        ./pentago -d 17 # Prove that depth 17 is a tie (takes many hours)

2. endgame: Out-of-core endgame database computation.  The state space is
   divided into "sections" consisting of positions with the same number of
   each color stone in each quadrant.  Depending on symmetries, each section
   depends on up to four child sections with one additional stone.  None of
   the files necessarily fit in RAM, so they are streamed from and to disk
   during the computation.  In addition to the large `section-*-.pentago`
   file containing the full data, endgame produces a sparse random sample
   file named `sparse-*-.try` for testing purposes.

        ./endgame --recurse 1 44444444 # Recursively compute section 44444444 and all its children

   By default, all files are written into ./data.  Use `--dir` to change this.

3. analyze: Compute various statistics and estimates about the game

        ./analyze approx # Print statistics for each level of the endgame traversal

4. draw-history: Visualize a history file written with the `--history <file>` option to endgame.

        ./endgame --history history.try ...
        ./draw-history history.try

5. tensor-info: Summarize a .pentago file

6. filter-test: Benchmark various filtering and compression schemes against each other

7. learn: Obsolete script generating statistics from a large list of random games.

8. opening: Obsolete opening book computation

### Code overview

Here's a brief summary of each file in the code:

##### Utilities

1. sort: Indirect insertion sort
2. thread: Utilities for job management via thread pools (a thin layer on top of pthreads)
3. stat: Fast statistics collection for use in the forward solvers
4. ashelve.py: Obsolete atomic python shelf for use in the (obsolete) opening book computation
5. convert: Utility file exposing different types of arrays to python

##### Standard and simple forward solvers:

1. board: Pentago boards are packed in 64-bit integers, using 16 bits for each quadrant as
   a 9 digit radix 3 integer.  Utilities are provided for breaking 64-bit boards into their
   component quadrants, or switching from radix 3 to 2 (a bitmask for one side) using
   lookup tables.

2. moves: Move generation code using macros and variable-length arrays to allocate memory
   only on the heap, for use in the forward solvers. 

3. score: Check whether a given position is a win for either player, or bound the number of
   moves required for a win for use as a search cutoff (win-distance pruning).

4. table: Hashed transposition table.  Since we use a 64-bit invertible has function, each
   entry need only store the high bits of the hash left out of the array index.

5. engine: Two different forward solvers operating on single positions at a time.  The first
   (normal) solver recursed over the full branching factor of pentago, which starts out in
   the 200s.  As a consequence, exhaustive search could only reach depth 4 or so.  Believing
   that the game was a strong tie, I wrote a second "simple" solver which handicapped white's
   (the second player's) choices by (1) always having white reverse black's last rotation,
   and (2) ignoring any white five-in-a-rows.  This avoided the need to branch on rotation:
   the `rotated_won` function in score simultaneously checks whether a win is possible if
   at most one of the quadrants is rotated.

   Unfortunately, the simple solver was too much of a simplification: black wins the modified
   game on move 15, establishing nothing about the real game.  It was a useful exercise,
   though, so it showed the speedups possible by dodging the rotation branching factor.

##### "Super" forward solver:

The "superengine" is essentially a standard alpha-beta tree search engine, but operating on
the lattice {0,1}^256 representing win flags for each the 256 possible boards given by rotating
a given start board.

1. superscore: Definition of the basic type `super_t`, storing 256 bits as 2 `__m128i`'s,
   together with a lookup table-based routine mapping a board to the super\_t of which of
   its rotations are wins.  Another key routine is `rmax`, which computes a bitwise or over
   each the 8 possible single rotations.  `rmax` is used to simulate branching over rotation
   without actually branching.

2. symmetry: In addition to avoid rotation branching, abstracting over rotations has the advantage
   of raising the size of the symmetry group from 8 to 2048.  An element of this group is
   stored as the type `symmetry_t` and acts on boards and `super_t`'s using the various `transform_*`
   functions.  The smallest board equivalent to a given board can be found with `superstandardize`.

3. supertable: Symmetry-aware transposition tables.  Entries exist only in superstandardized form,
   and store two `super_t`'s: a mask of which entries we know and the values of those we do.  A bit
   of logic is required when combining values of different depths together: information of low depth
   can be used if it signified an immediate end to the game, otherwise not.

4. superengine: The main rotation abstracted forward solver.  It is structurally similar to the
   "simple" solver, including the use of pruning and move ordering based on bounds on closeness
   to the nearest win, but operates on the full game rather than an approximate version.  As
   mentioned above, starting from an empty board, this solver takes about 8.2 hours on a 2.3
   GHz Macbook Pro to prove that pentago is a tie within the first 17 moves.

5. trace: Optional debugging code used to track down bugs in the superengine.

##### Enumeration and counting

1. count: Exact counting of the number of symmetry-reduce pentago positions using the
   Polya enumeration theorem.

2. all\_boards: Explicit enumeration of symmetry-reduced pentago positions with a given
   number of stones.  Also contains routines for enumerating "sections", describing all
   stones with a given number of stones in each quadrant.

3. analyze.cpp: Helper routines for the `analyze` script.

##### Endgame (backward) solver

1. filter: Various routines to precondition win/loss/tie data for input to zlib/xz.
   Sadly, the complicated ones didn't seem promising enough to implement, so the currently
   used default is `interleave`, which simply transposes a pair of `super_t`'s into 256 2-bit
   pairs.

2. supertensor: A supertensor file is 4 dimensional array of `Vector<super_t,2>` giving
   win/loss/tie values for all positions in a given section.  Since a section is defined
   by the number of black and white stones in each quadrant, the four quadrants define the
   four dimensions of an array.  The largest supertensor files consume up to a terabyte
   uncompressed, so we perform blocking along all four dimensions with a block size small
   enough that an entire two dimensional block slice fits easily into RAM.  The 4-symmetry
   of the format is critical, since during computation each section is first standardized
   (rotated and reflected into the minimal configuration), and thus the dimensions on disk
   are often a permuted version of what we actually need.

3. endgame.cpp: Read/compute/write routines for block slices of supertensor files.
   Parallelized using the thread pools defined in `thread.h`.

4. endgame: Python driver routine for endgame computation.

##### Precomputation

Many of the above files depend on precomputated tables generated during the build process
by `precompute.py`.  These include tables for rotated or reflecting a quadrant, helper tables
for detecting wins or win proximity, tables to switch between radix 2 and 3, tables to help
with symmetry operations, and enumeration of rotation minimal quadrants in special orderings.
