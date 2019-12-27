// Solve positions near the end of the game using downstream retrograde analysis
//
// If we fix a given 18 stone board, there are 18 free spots.  Since we abstract
// away rotations, these free spots do not move around, so the set of positions
// can be compactly indexed using combinatorial trickery.  This allows efficient
// traversal of exactly those position downstream of a given root.  For boards with
// 18 stones, this takes about 15 seconds on a Rackspace instance using less than
// 1 GB of RAM.  Conveniently, the time depends solely on the number of stones, not
// the particular board, which is perfectly suited for use as a web service.
#pragma once

#include "pentago/mid/halfsuper.h"
#include "pentago/high/board.h"
#include "pentago/utility/pile.h"
#include <tuple>
namespace pentago {

using std::tuple;

// (win,notlose)
typedef Vector<halfsuper_t,2> halfsupers_t;
struct superinfos_t {
  super_t known, win, notlose;
};

// A k-subsets of [0,n-1], packed into 64-bit ints with 5 bits for each entry.
typedef uint64_t set_t;
void subsets(const int n, const int k, RawArray<set_t> sets);

// Determine workspace memory required for midsolve.
// Use midsolve_workspace if you're in C++; this one is for Javascript.
uint64_t midsolve_workspace_memory_usage(const int min_slice);

#ifndef __wasm__
// Allocate enough memory for midsolves with at least the given number of stones
Array<uint8_t> midsolve_workspace(const int min_slice);
#endif  // !__wasm__

typedef pile<tuple<high_board_t,int>,1+18+8*18> midsolve_results_t;
typedef pile<tuple<board_t,superinfos_t>,1+18> midsolve_internal_results_t;

// Compute the values of a board and its children, assuming the board has at least 18 stones.
midsolve_internal_results_t midsolve_internal(const board_t root, const bool parity, RawArray<uint8_t> workspace);

// Compute the values of a board, its children, and possibly children's children (if !board.middle)
midsolve_results_t midsolve(const high_board_t board, RawArray<uint8_t> workspace);

}
