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
NAMESPACE_PENTAGO

using std::tuple;

// (win,notlose)
typedef Vector<halfsuper_t,2> halfsupers_t;
struct superinfos_t {
  halfsuper_t win, notlose;
  bool parity;

  bool known(const int s) const { return ((s^s>>2^s>>4^s>>6) & 1) == parity; }
  int value(const int s) const { return win[s >> 1] + notlose[s >> 1] - 1; }
};

// A k-subsets of [0,n-1], packed into 64-bit ints with 5 bits for each entry.
typedef uint64_t set_t;
void subsets(const int n, const int k, RawArray<set_t> sets);

// Size of workspace array needed by midsolve
int midsolve_workspace_size(const int min_slice);

#ifndef __wasm__
// Allocate enough memory for midsolves with at least the given number of stones
Array<halfsupers_t> midsolve_workspace(const int min_slice);
#endif  // !__wasm__

struct mid_values_t : pile<tuple<high_board_t,int>,1+18+8*18> {};
struct mid_supers_t : pile<tuple<Vector<side_t,2>,superinfos_t>,1+18> {};

// Compute the values of a board and its children, assuming the board has at least 18 stones.
mid_supers_t midsolve_internal(const high_board_t root, RawArray<halfsupers_t> workspace);

#if !defined(__wasm__) || defined(__APPLE__)
// Compute the values of a board, its children, and possibly children's children (if !board.middle)
mid_values_t midsolve(const high_board_t board, RawArray<halfsupers_t> workspace);
#endif

END_NAMESPACE_PENTAGO
