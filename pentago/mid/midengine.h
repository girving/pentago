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

#include "halfsuper.h"
#include "../high/board.h"
#include "../utility/pile.h"
#include <tuple>
NAMESPACE_PENTAGO

using std::tuple;

// Size of workspace array needed by midsolve
int midsolve_workspace_size(const int min_slice);

#ifndef __wasm__
// Allocate enough memory for midsolves with at least the given number of stones
Array<halfsuper_s> midsolve_workspace(const int min_slice);
#endif  // !__wasm__

static inline int value(const halfsupers_t& I, const int s) {
  return get(I.win, s >> 1) + get(I.notlose, s >> 1) - 1;
}

static inline int mid_supers_size(const high_board_t board) {
  return 1 + (36 - board.count());
}

struct mid_values_t : pile<tuple<high_board_t,int>,1+18+8*18> {};

// Compute the values of a board and its children, assuming the board has at least 18 stones.
// Results are {whether we don't lose, whether we win}
Vector<halfsupers_t,1+18> midsolve_internal(const high_board_t root, RawArray<halfsuper_s> workspace);
int midsolve_traverse(const high_board_t board, const halfsuper_s* wins, const halfsuper_s* notloses, mid_values_t& results);

#if !defined(__wasm__) || defined(__APPLE__)
// Compute the values of a board, its children, and possibly children's children (if !board.middle)
mid_values_t midsolve(const high_board_t board, RawArray<halfsuper_s> workspace);
#endif

END_NAMESPACE_PENTAGO
