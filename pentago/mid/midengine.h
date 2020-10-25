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

#include "pentago/mid/midengine_c.h"
#include "pentago/mid/halfsuper.h"
#include "pentago/high/board.h"
#include "pentago/utility/pile.h"
#include <tuple>
NAMESPACE_PENTAGO

using std::tuple;

// Size of workspace array needed by midsolve
int midsolve_workspace_size(const int min_slice);

#ifndef __wasm__
// Allocate enough memory for midsolves with at least the given number of stones
Array<halfsupers_t> midsolve_workspace(const int min_slice);
#endif  // !__wasm__

static inline bool known(const superinfos_t& I, const int s) {
  return ((s^s>>2^s>>4^s>>6) & 1) == I.parity;
}

static inline int value(const superinfos_t& I, const int s) {
  return get(I.win, s >> 1) + get(I.notlose, s >> 1) - 1;
}

static inline int mid_supers_size(const high_board_t board) {
  return 1 + (36 - board.count());
}

struct mid_values_t : pile<tuple<high_board_t,int>,1+18+8*18> {};

// Compute the values of a board and its children, assuming the board has at least 18 stones.
Vector<mid_super_t,1+18> midsolve_internal(const high_board_t root, RawArray<halfsupers_t> workspace);
void midsolve_traverse(const high_board_t board, const mid_super_t* supers, mid_values_t& results);

#if !defined(__wasm__) || defined(__APPLE__)
// Compute the values of a board, its children, and possibly children's children (if !board.middle)
mid_values_t midsolve(const high_board_t board, RawArray<halfsupers_t> workspace);
#endif

END_NAMESPACE_PENTAGO
