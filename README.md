A massively parallel pentago solver
===================================

This project implements a strong solution of the board game pentago, which means
that perfect play can be efficiently computed starting from any position.  The
results can be explored at https://perfect-pentago.net.  The associated paper
is <a href="https://arxiv.org/abs/1404.0743">Irving 2014, Pentago is a first-player win</a>.

In the quest towards a strong solution, a total of five separate engines were
implemented, three forward tree search engines and two backwards endgame engines.
The final strong solution uses only the backwards engines, but the strongest
forward engine was essential in developing and testing the backwards engines.
Descriptions of the various engines are given below.

The easiest interface to the strong solution is the website at
https://perfect-pentago.net, implemented in the `web` directory.  Positions with 17
or fewer stones are looked up in a 3.7 TB database using a minimal node.js server
(`web/server`).  Positions with 18 or more stones are solved from scratch in the
client, using small-scale backwards engine compiled to WebAssembly.

The pentago code is released under a BSD license, and the 3.7 TB data set is
released into the public domain for anyone who wants to tinker around: please contact
me if you want access to the raw data!

For background on the complexity of pentago and the applicability of a variety of
solution algorithms, see Niklas Buscher's 2011 thesis
["On Solving Pentago"](http://www.ke.tu-darmstadt.de/lehre/arbeiten/bachelor/2011/Buescher_Niklas.pdf),
which analyzed the game but did not solve it.
<a href="https://arxiv.org/abs/1404.0743">Irving 2014</a> documents the strong solution
in paper form.

### Dependencies

The code is C++, with node.js for the backend server.  The main dependencies are

* [bazel](https://bazel.build): Build system
* [mpi](http://en.wikipedia.org/wiki/Message_Passing_Interface): OpenMPI, MPICH2, etc.
* [node.js >= 8.9](http://nodejs.org): Asynchronous javascript framework

Bazel handles a few extra dependencies automatically (see `WORKSPACE` for details).

### Installation

On Mac:

    # Install dependencies
    brew install bazel openmpi node llvm

    # Build and test C++
    bazel test -c opt --copt=-march=native ...

    # Build and test node.js server
    cd web/server
    npm install
    node unit.js all

    # Build frontend webpage
    cd web/client
    make public

On Ubuntu: to be written once I have clean Bazel handling of MPI.

### Website

The website https://perfect-pentago.net is a static Firebase frontend that
talks to a node.js Google Cloud Function.  To test and deploy the server:

    cd web/server
    node unit.js all
    ./deploy

To deploy the client:

    cd pentago/web/client
    npm install
    make
    (cd src && node unit.js)
    npm run deploy

    # If there is an error like 'resolving hosting target of a site with no site name...', do
    firebase logout
    firebase login

## Algorithm summary

### Forward engines

The first forward engine was a naive alpha-beta tree search code, plus a few pentago
specific optimizations.  Performance was dismal: the code reached only to 4 or 5 ply
(here one ply means the combination of stone placement and quadrant rotation).
The problem was branching factor, which starts out at a frightening 288 possible
first moves (36 empty spots times 8 rotations).

Based on a belief that pentago was likely a tie (wrong!), I modified the solver so
that the second player always reversed the rotation of the first player.  This can't
be any better for the second player, so if the result was a tie the game would be
solved.  Unfortunately, this "simple" solver declared the game to be a first-player
win after 15 moves, providing no information about the real value of the game.

The key to the speed of the simple solver was eliminating the branching factor due
to rotations.  The final forward solver accomplished the same thing without weakening
the second player.  Instead of computing the value of one board at a time, this
"super" solver operated on 256 different rotated version of a given board in parallel
using SSE instructions.  This also allowed the transposition tables to take advantage
of the full 2048 element symmetry group of all global+local transpositions.  With
rotations eliminated, the supersolver managed to compute out to 17 ply, declaring the
game up to that point a tie.  See `pentago/search/superengine.{h,cpp}` for the core
algorithm.

Parallelizing the engine would get further, but I still thought the
game was a tie, and no forward engine would reach all the way to move 36.  The similar
games gomoku and renju were solved using threat space search, but it was unclear how
to combine this technique with a rotation abstracted solver.  So much for going forward.

### A massively parallel backward engine

What about backward?  Typically, backward (or retrograde) analyses are used to solve
convergent games, where the number of positions decreases towards the end of the game.
The canonical examples are chess and checkers, where fewer pieces near the end mean
fewer positions.  Pentago is divergent: the number of positions increases exponentially
all the way to ply 24, but the total number of positions is only $3 \times 10^{15}$.

Unfortunately, the computation required 80 TB of memory at peak.  Since the arithmetic
intensity of computing pentago positions is low, shuffling all of this data in
and out of disks would have destroyed performance.  Therefore, the endgame solver was
implemented in-core, parallelized using MPI for use on a supercomputer.
Like the final forward solver, the endgame solver takes advantage of symmetry
by operating on 256 positions at a time and eliminating symmetric positions.  The code
for the massively parallel endgame solver is contained in `pentago/end` and `pentago/mpi`.

This backward engine was sufficient to strongly solve the game, requiring
about four hours of time on 98304 threads of Edison, a Cray supercomputer at NERSC.

Like endgame solvers for other games, the backwards engine starts at the end of the game
and walks backwards towards the beginning, computing the value of each position.  Unlike
previous endgame solvers, nearly all of the data is discarded once it is used, and only
positions with 18 or fewer stones are kept and written to disk.  To my knowledge, this is
the first use of an endgame solver to directly compute an opening book for a nontrivial game.

### A tiny backward engine

Since the massively parallel computation discarded all positions with more than 18 stones,
the game isn't strongly solved unless the missing values can be quickly recomputed.  Before
the massively parallel computation, I tried the forward solver on a variety of 18 stone
positions and got satisfactory performance: the solver was capable of reaching forwards to
the end of the game.  This was naive: random positions are much easier than
interesting positions, and I quickly found interesting positions once I began exploring the
results.  The first time I came across one of these the solve took an hour; I managed to
get it down to 10 minutes, but this was still much too slow for an interactive website.

I thus implemented another backward engine, this one specialized to
traverse only positions downstream of a given root board with 18 or more stones.  In addition to
tackling a much smaller problem than the massively parallel engine, this "midgame" engine
is also much simpler: rotations are abstracted away, so the set of empty positions is
fixed by the root board.  As a perk, we can use parity to operate on only 128 different
rotated positions instead of 256, as rotating the board flips the parity of the local
rotation group.  On my M2 MacBook Air, the midgame engine takes around 3.1 s in C++, and 8.2 s
using WebAssembly for *any* 18 stone position; unlike the forward solver performance does not
vary with the board.  The code for the midgame engine is in `pentago/mid`.

Together, the 3.7 TB data set and the midgame engine comprise a strong solution of pentago.
Go to https://perfect-pentago.net to see the results.

### Code structure

The code is organized into the following directories:

* `pentago/utility`: Thread pools, memory allocation, and other non-pentago-specific utilities
* `pentago/base`: Pentago boards, moves, scores, symmetries, counts, etc., used by the various solvers.
* `pentago/search`: Forward search engines
* `pentago/data`: File formats and I/O for the backwards engine
* `pentago/end`: Everything in the backwards engine that doesn't directly touch mpi
* `pentago/mpi`: Everything in the backwards engine that does touch mpi
* `pentago/high`: High level interface used in the website backend
* `pentago/mid`: "Midgame" backward solver used to complete the strong solution for 18 or more stone boards
* `web/server`: <a href="https://nodejs.org">node.js</a> lookup server for access to the 3.7 TB database
* `web/client`: <a href="https://svelte.dev">svelte</a> + <a href="https://webassembly.org">wasm</a> client

Most files have summarizing comments at the top, but otherwise documentation is scant.
Email me if you have questions at <irving@naml.us>!

### Data

The total computed database size is 3.7 TB, using custom formats accessible using the code.
[`data/edison/final.txt`](https://github.com/girving/pentago/tree/master/data/edison/final.txt) shows the
list of files and sizes.  There are three types of files:

* `slice-{0...18}.pentago`: Full data for positions with at most 18 stones
* `sparse-{0...36}.pentago`: A sparse subsample of positions with any number of stones
* `counts-{0...36}.pentago`: Counts of positions that each player wins, at each number of stones

The raw data is freely available on request: please email <irving@naml.us> if you want it.
