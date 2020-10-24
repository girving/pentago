// Internal routines for midengine.cc and metal equivalent
#pragma once

#include "pentago/mid/subsets.h"
#include "pentago/base/board.h"
#include "pentago/high/board.h"
#include <stdbool.h>
#ifdef __cplusplus
#include "pentago/utility/integer_log.h"
#endif  // __cplusplus
NAMESPACE_PENTAGO

typedef struct grab_t_ {
  int nx, ny;
  int lo, hi;
} grab_t;

// Constant information for midsolve_loop
typedef struct info_t_ {
  high_board_s root;
  int n, parity, slice, spots, k0, k1;
  bool done;
  side_t root0, root1;
  sets_t sets0, sets1, sets1p, sets0_next;
  int cs1ps_size;
  empty_t empty;
  grab_t input, output;
} info_t;

typedef struct wins_info_t_ {
  empty_t empty;
  side_t root0, root1;
  sets_t sets1, sets0_next;
  int size;
  bool parity;
} wins_info_t;

// Everything that's a function of just s0 in the double loop in midsolve_loop
typedef struct set0_info_t_ {
  set_t set0;
  side_t side0;
  int s0;
  uint32_t filled0;
  halfsuper_t wins0;
  halfsuper_t child_wins0[18];
  uint8_t empty1[18];
  uint16_t child_s0s[18];
  uint16_t offset1[10][18];
  uint16_t offset0[10][18];
} set0_info_t;

_Static_assert(sizeof(set0_info_t) >= 1104, "");
_Static_assert(sizeof(set0_info_t) <= 1120, "");

#ifdef __cplusplus

static inline grab_t make_grab(const bool end, const int nx, const int ny, const int workspace_size) {
  grab_t g;
  g.nx = nx;
  g.ny = ny;
  const auto r = end ? range(workspace_size-nx*ny, workspace_size) : range(0,nx*ny);
  g.lo = r.lo;
  g.hi = r.hi;
  return g;
}

static inline RawArray<halfsupers_t,2> slice(RawArray<halfsupers_t> workspace, const grab_t& g) {
  return workspace.slice(g.lo, g.hi).reshape(vec(g.nx, g.ny));
}

static inline info_t make_info(const high_board_t root, const int n, const int workspace_size) {
  info_t I;
  I.root = root;
  I.n = n;
  I.parity = root.middle();
  I.slice = root.count();
  I.spots = 36 - I.slice;
  I.k0 = (I.slice+n)/2 - (I.slice+(n&1))/2;
  I.k1 = n - I.k0;
  I.done = I.spots == I.n;
  I.root0 = root.side((I.slice+n)&1);
  I.root1 = root.side((I.slice+n+1)&1);
  I.sets0 = make_sets(I.spots, I.k0);
  I.sets1 = make_sets(I.spots, I.k1);
  I.sets1p = make_sets(I.spots-I.k0, I.k1);
  I.sets0_next = I.done ? make_sets(0, 1) : make_sets(I.spots, I.k0+1);
  I.cs1ps_size = I.sets1p.size * (I.spots-I.k0);
  init(I.empty, root);
  I.input = make_grab(n&1, choose(I.spots, I.k0+1), choose(I.spots-I.k0-1, I.k1), workspace_size);
  I.output = make_grab(!(n&1), I.sets1.size, choose(I.spots-I.k1, I.k0), workspace_size);
  return I;
}

static inline wins_info_t make_wins_info(const info_t& I) {
  wins_info_t W;
  W.empty = I.empty;
  W.root0 = I.root0;
  W.root1 = I.root1;
  W.sets1 = I.sets1;
  W.sets0_next = I.sets0_next;
  W.size = W.sets1.size + W.sets0_next.size;
  W.parity = (I.n+I.parity)&1;
  return W;
}

// [:sets1.size] = all_wins1 = whether the other side wins for all possible sides
// [sets1.size:] = all_wins0_next = whether we win on the next move
static inline halfsuper_t mid_wins(const wins_info_t& W, const int s) {
  const bool one = s < W.sets1.size;
  const auto ss = one ? s : s - W.sets1.size;
  const auto root = one ? W.root1 : W.root0;
  const auto sets = one ? W.sets1 : W.sets0_next;
  return halfsuper_wins(root | side(W.empty, sets, ss), W.parity);
}

static inline uint16_t make_cs1ps(const info_t& I, const set_t* sets1p, const int index) {
  const auto [s1p, j] = decompose(vec(I.sets1p.size, I.spots-I.k0), index);
  uint16_t c = s1p;
  for (const int a : range(I.k1)) {
    const int s1p_a = sets1p[s1p]>>5*a&0x1f;
    if (j<s1p_a)
      c += fast_choose(s1p_a-1, a+1) - fast_choose(s1p_a, a+1);
  }
  return c;
}

static inline set0_info_t make_set0_info(const info_t& I, const halfsuper_t* all_wins, const int s0) {
  set0_info_t I0;
  const int k0 = I.sets0.k;
  const int k1 = I.n - I.k0;

  // Construct side to move
  I0.s0 = s0;
  I0.set0 = get(I.sets0, s0);
  I0.side0 = I.root0 | side(I.empty, I.sets0, I0.set0);

  // Make a mask of filled spots
  I0.filled0 = 0;
  for (const int i : range(k0))
    I0.filled0 |= 1<<(I0.set0>>5*i&0x1f);

  // Evaluate whether we win for side0 and all child sides
  I0.wins0 = halfsuper_wins(I0.side0, (I.n+I.parity)&1);

  // List empty spots after we place s0's stones
  {
    const auto free = side_mask & ~I0.side0;
    int next = 0;
    for (const int i : range(I.spots))
      if (free&side_t(1)<<I.empty.empty[i])
        I0.empty1[next++] = i;
    NON_WASM_ASSERT(next==I.spots-I.k0);
  }

  /* What happens to indices when we add a stone?  We start with
   *   s0 = sum_i choose(set0[i],i+1)
   * at q with set0[j-1] < q < set0[j], the new index is
   *   cs0 = s0 + choose(q,j+1) + sum_{j<=i<k0} [ choose(set0[i],i+2)-choose(set0[i],i+1) ]
   *
   * What if we instead add a first white stone at empty1[q0] with set0[j0-1] < empty1[q0] < set0[j0]>
   * The new index is s0p_1:
   *   s0p_0 = s0
   *   s0p_1 = s0p_0 + sum_{j0<=i<k0} choose(set0[i]-1,i+1)-choose(set0[i],i+1)
   *         = s0p_0 + offset0[0][q0]
   * That is, all combinations from j0 on are shifted downwards by 1, and the total offset is precomputable.
   * If we add another stone at q1,j1, the shift is
   *   s0p_2 = s0p_1 + sum_{j1<=i<k0} choose(set0[i]-2,i+1)-choose(set0[i]-1,i+1)
   *         = s0p_1 + offset0[1][q1]
   * where we have used the fact that q0 < q1.  In general, we have
   *   s0p_a = s0p_{a-1} + sum_{ja<=i<k0} choose(set0[i]-a-1,i+1)-choose(set0[i]-a,i+1)
   *         = s0p_{a-1} + offset0[a][qa]
   *
   * So far this offset0 array is two dimensional, but now we have to take into account the new black
   * stones.  We must also be able to know where they go.  The easy thing is to make one 2D offset0 array
   * for each place we can put the black stone, but that requires us to do independent sums for every
   * possible move.  It seems difficult to beat, though, since the terms in the sum that makes up cs0p-s0p
   * can change independently as we add white stones.
   *
   * Overall, this precomputation seems quite complicated for an uncertain benefit, so I'll put it aside for now.
   *
   * ----------------------
   *
   * What if we start with s1p and add one black stone at empty1[i] to reach cs1p?  We get
   *
   *    s1p = sum_j choose(set1p[j],j+1)
   *   cs1p = sum_j choose(set1p[j]-(set1p[j]>i),j+1)
   *   cs1p = s1p + sum_{j s.t. set1p[j]>i} choose(set1p[j]-1,j+1) - choose(set1p[j],j+1)
   *
   * Ooh: that's complicated, but it doesn't depend at all on s0, so we can hoist the entire
   * thing.  See cs1ps above.
   */

  // Precompute absolute indices after we place s0's stones
  for (const int i : range(I.spots-I.k0)) {
    const int j = I0.empty1[i]-i;
    I0.child_s0s[i] = choose(I0.empty1[i], j+1);
    for (const int a : range(k0))
      I0.child_s0s[i] += choose(I0.set0>>5*a&0x1f, a+(a>=j)+1);
  }

  // Preload wins after we place s0's stones.
  if (!I.done) {
    const auto all_wins0_next = all_wins + I.sets1.size;
    for (const int i : range(I.spots-I.k0))
      I0.child_wins0[i] = all_wins0_next[I0.child_s0s[i]];
  }

  // Lookup table to convert s1p to s1
  for (const int i : range(k1))
    for (const int q : range(I.spots-I.k0))
      I0.offset1[i][q] = fast_choose(I0.empty1[q], i+1);

  // Lookup table to convert s0 to s0p
  for (const int a : range(k1)) {
    for (const int q : range(I.spots-I.k0)) {
      I0.offset0[a][q] = 0;
      for (int i = I0.empty1[q]-q; i < I.k0; i++) {
        const int v = I0.set0>>5*i&0x1f;
        if (v>a)
          I0.offset0[a][q] += fast_choose(v-a-1, i+1) - fast_choose(v-a, i+1);
      }
    }
  }
  return I0;
}

static inline void inner(const info_t& I, const uint16_t* cs1ps, RawArray<const set_t> sets1p,
                         const halfsuper_t* all_wins, mid_super_t* results, RawArray<halfsupers_t> workspace,
                         const set0_info_t& I0, const int s1p) {
  const auto set1p = sets1p[s1p];
  const auto input = slice(workspace, I.input).const_();
  const auto output = slice(workspace, I.output);
  const auto all_wins1 = all_wins;  // all_wins1 is a prefix of all_wins

  // Convert indices
  uint32_t filled1 = 0;
  uint32_t filled1p = 0;
  uint16_t s1 = 0;
  uint16_t s0p = I0.s0;
  for (const int i : range(I.k1)) {
    const int q = set1p>>5*i&0x1f;
    filled1 |= 1<<I0.empty1[q];
    filled1p |= 1<<q;
    s1  += I0.offset1[i][q];
    s0p += I0.offset0[i][q];
  }

  // Consider each move in turn
  halfsupers_t us;
  {
    us[0] = us[1] = halfsuper_t(0);
    if (I.slice + I.n == 36)
      us[1] = ~halfsuper_t(0);
    auto unmoved = ~filled1p;
    for (int m = 0; m < I.spots-I.k0-I.k1; m++) {
      const auto bit = min_bit(unmoved);
      const int i = integer_log_exact(bit);
      unmoved &= ~bit;
      const int cs0 = I0.child_s0s[i];
      const auto cwins = I0.child_wins0[i];
      const halfsupers_t child = input(cs0, cs1ps[s1p*(I.spots-I.k0)+i]);
      us[0] |= cwins | child[0];  // win
      us[1] |= cwins | child[1];  // not lose
    }
  }

  // Account for immediate results
  const halfsuper_t wins1 = all_wins1[s1],
                    inplay = ~(I0.wins0 | wins1);
  us[0] = (inplay & us[0]) | (I0.wins0 & ~wins1);  // win (| ~inplay & (I0.wins0 & ~wins1))
  us[1] = (inplay & us[1]) | I0.wins0;  // not lose       (| ~inplay & (I0.wins0 | ~wins1))

  // If we're far enough along, remember results
  if (I.n <= 1) {
    const uint32_t filled_black = (I.slice+I.n)&1 ? filled1 : I0.filled0,
                   filled_white = (I.slice+I.n)&1 ? I0.filled0 : filled1;
    auto side0 = I.root.side_[0];
    auto side1 = I.root.side_[1];
    for (const int i : range(I.spots)) {
      side0 |= side_t(filled_black>>i&1) << I.empty.empty[i];
      side1 |= side_t(filled_white>>i&1) << I.empty.empty[i];
    }
    auto& r = results[I.n + s1p];
    r.sides[0] = side0;
    r.sides[1] = side1;
    r.supers = superinfos_t{us[0], us[1], bool((I.n+I.parity)&1)};
  }

  // Negate and apply rmax in preparation for the slice above
  halfsupers_t above;
  above[0] = rmax(~us[1]);
  above[1] = rmax(~us[0]);
  output(s1,s0p) = above;
}

#endif  // __cplusplus
END_NAMESPACE_PENTAGO
