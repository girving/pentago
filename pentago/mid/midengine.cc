// Solve positions near the end of the game using downstream retrograde analysis

#include "pentago/mid/midengine.h"
#include "pentago/mid/internal.h"
#include "pentago/mid/subsets.h"
#include "pentago/base/symmetry.h"
#include "pentago/utility/ceil_div.h"
#include "pentago/utility/debug.h"
#include "pentago/utility/integer_log.h"
#include "pentago/utility/wasm_alloc.h"
#ifndef __wasm__
#include "pentago/utility/aligned.h"
#include "pentago/utility/log.h"
#endif  // !__wasm__
NAMESPACE_PENTAGO

using std::max;
using std::make_tuple;

/* In the code below, we will organized pairs of subsets of [0,n-1] into two dimensional arrays.
 * The first dimension records the player to move, the second the other player.  The first dimension
 * is organized according to the order produced by subsets(), and the second is organized by the
 * order produced by subsets *relative* to the first set (a sort of semi-direct product).  For example,
 * if our root position has slice 32, or 4 empty spots, the layout is
 *
 *   n 0 : ____
 *   n 1 : 0___  _0__  __0_ ___0
 *
 *   n 2 : 01__  0_1_  0__1
 *         10__  _01_  _0_1
 *         1_0_  _10_  __01
 *         1__0  _1_0  __10
 *
 *   n 3 : 100_  10_0  1_00
 *         010_  01_0  _100
 *         001_  0_10  _010
 *         00_1  0_01  _001
 *
 *   n 4 : 0011
 *         0101
 *         1001
 *         0110
 *         1010
 *         1100
 */

static int bottleneck(const int spots) {
  int worst = 0;
  int prev = 1;
  for (const int n : range(spots+1)) {
    int next = choose(spots, n+1);
    if (next) next *= choose(n+1, (n+1)/2);
    worst = max(worst, prev + next);
    prev = next;
  }
  return worst;
}

int midsolve_workspace_size(const int min_slice) {
  return bottleneck(36 - min_slice);
}

#ifndef __wasm__
Array<halfsupers_t> midsolve_workspace(const int min_slice) {
  return aligned_buffer<halfsupers_t>(midsolve_workspace_size(min_slice));
}
#endif  // !__wasm__

static void midsolve_loop(const high_board_t root, const int n, mid_super_t* results,
                          RawArray<halfsupers_t> workspace) {
  const info_t I = make_info(root, n, workspace.size());

  // Precompute subsets of player 1 relative to player 0's stones
  set_t sets1p[I.sets1p.size];
  for (const int s1p : range(I.sets1p.size))
    sets1p[s1p] = get(I.sets1p, s1p);

  // Precompute various halfsuper wins
  const auto W = make_wins_info(I);
  halfsuper_t all_wins[W.size];
  for (const int s : range(W.size))
    all_wins[s] = mid_wins(W, s);

  // Lookup table for converting s1p to cs1p (s1 relative to one more black stone):
  //   cs1p = cs1ps[s1p].x[j] if we place a black stone at empty1[j]
  uint16_t cs1ps[I.cs1ps_size];
  for (const int i : range(I.cs1ps_size))
    cs1ps[i] = make_cs1ps(I, sets1p, i);

  // Iterate over set of stones of player to move
  for (const int s0 : range(I.sets0.size)) {
    const set0_info_t I0 = make_set0_info(I, all_wins, s0);
    // Iterate over set of stones of other player
    for (const int s1p :  range(I.sets1p.size))
      inner(I, cs1ps, sets1p, all_wins, results, workspace, I0, s1p);
  }
}

Vector<mid_super_t,1+18> midsolve_internal(const high_board_t board, RawArray<halfsupers_t> workspace) {
  const int spots = 36 - board.count();
  NON_WASM_ASSERT(workspace.size() >= bottleneck(spots));

  // Compute all slices
  Vector<mid_super_t,1+18> results;
  for (int n = spots; n >= 0; n--)
    midsolve_loop(board, n, results.data(), workspace);
  return results;
}

static int traverse(const high_board_t board, RawArray<const mid_super_t> supers, mid_values_t& results) {
  int value;
  const auto [done, immediate_value] = board.done_and_value();
  if (done) { // Done, so no lookup required
    value = immediate_value;
  } else if (!board.middle()) {  // Recurse into children
    value = -1;
    const auto empty = board.empty_mask();
    for (const int bit : range(64))
      if (empty & side_t(1)<<bit)
        value = max(value, traverse(board.place(bit), supers, results));
  } else {  // if board.middle()
    // Find unrotated board in supers
    int i;
    for (i = 0; i < supers.size(); i++) {
      const side_t* sides = supers[i].sides;
      if (sides[0] == board.side(0) && sides[1] == board.side(1))
        break;
    }
    NON_WASM_ASSERT(i < supers.size());
    const superinfos_t& r = supers[i].supers;

    // Handle recursion manually to avoid actual rotation
    value = -1;
    for (const int s : range(8)) {
      const int q = s >> 1;
      const int d = s & 1 ? -1 : 1;
      const auto v = r.value((d & 3) << 2*q);
      results.append(make_tuple(board.rotate(q, d), v));
      value = max(value, -v);
    }
  }

  // Store and return value
  results.append(make_tuple(board, value));
  return value;
}

void midsolve_traverse(const high_board_t board, const mid_super_t* supers, mid_values_t& results) {
  const auto prefix = RawArray<const mid_super_t>(mid_supers_size(board), supers);
  traverse(board, prefix, results);
}

#if !defined(__wasm__) || defined(__APPLE__)
mid_values_t midsolve(const high_board_t board, RawArray<halfsupers_t> workspace) {
  // Compute
  const auto supers = midsolve_internal(board, workspace);

  // Extract all available boards
  mid_values_t results;
  midsolve_traverse(board, supers.data(), results);
  return results;
}
#else  // if __wasm__
WASM_EXPORT void midsolve(const high_board_t* board, mid_values_t* results) {
  NON_WASM_ASSERT(board && results);
  results->clear();
  const auto workspace = wasm_buffer<halfsupers_t>(midsolve_workspace_size(board->count()));
  const auto supers = midsolve_internal(*board, workspace);
  midsolve_traverse(*board, supers.data(), *results);
}
#endif  // __wasm__

END_NAMESPACE_PENTAGO
