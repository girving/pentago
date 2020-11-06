// Internal routines for midengine.cc and metal equivalent
#pragma once

#include "internal_c.h"
#include "midengine_c.h"
#include "halfsuper.h"
#include "subsets.h"
#include "../utility/integer_log.h"
NAMESPACE_PENTAGO

template<class info_ref=METAL_CONSTANT const info_t&> struct helper_t {
  info_ref I;
  const int n;

  bool done() const { return I.spots == n; }
  int k0() const { return n >> 1; }
  int k1() const { return n - k0(); }
  bool parity() const { return (n+I.root.ply_)&1; }
  side_t root0() const { return I.root.side_[(I.slice+n)&1]; }
  sets_t sets0() const { return make_sets(I.spots, k0()); }
  sets_t sets1() const { return make_sets(I.spots, k1()); }
  sets_t sets1p() const { return make_sets(I.spots-k0(), k1()); }
  sets_t fast_sets0_next() const { assert(!done()); return make_sets(I.spots, k0()+1); }
  sets_t sets0_next() const { return done() ? make_sets(0, 1) : fast_sets0_next(); }
  int sets1p_size() const { return I.sets1p_offsets[n+1] - I.sets1p_offsets[n]; }
  int wins_size() const { return I.wins_offsets[n+1] - I.wins_offsets[n]; }
  int cs1ps_size() const { return sets1p_size() * (I.spots-k0()); }
};

static inline grab_t make_grab(const bool end, const int nx, const int ny, const int workspace_size) {
  grab_t g;
  g.ny = ny;
  g.lo = end ? workspace_size - nx * ny : 0;
  g.size = nx * ny;
  return g;
}

// RawArray<halfsupers_t,2>, but minimal for Metal friendliness
struct io_t {
  METAL_DEVICE halfsupers_t* data;
  int stride;

  METAL_DEVICE halfsupers_t& operator()(const int i, const int j) const { return data[i * stride + j]; }
};

static inline io_t slice(METAL_DEVICE halfsupers_t* workspace, const grab_t g) {
  return io_t{workspace + g.lo, g.ny};
}

static inline info_t make_info(const high_board_t root, const int workspace_size) {
  info_t I;
  I.root = root.s;
  I.slice = root.count();
  I.spots = 36 - I.slice;
  I.empty = make_empty(root);
  I.sets0_offsets[0] = 0;
  I.sets1p_offsets[0] = 0;
  I.cs1ps_offsets[0] = 0;
  I.wins_offsets[0] = 0;
  for (int n = 0; n <= I.spots; n++) {
    const helper_t<METAL_THREAD const info_t&> H{I, n};
    I.sets0_offsets[n+1] = I.sets0_offsets[n] + H.sets0().size;
    I.sets1p_offsets[n+1] = I.sets1p_offsets[n] + H.sets1p().size;
    I.cs1ps_offsets[n+1] = I.cs1ps_offsets[n] + H.cs1ps_size();
    I.wins_offsets[n+1] = I.wins_offsets[n] + H.sets1().size + H.sets0_next().size;
    I.spaces[n] = make_grab(!(n&1), H.sets1().size, choose(I.spots-H.k1(), H.k0()), workspace_size);
  }
  I.spaces[I.spots+1] = make_grab(0, 0, 0, 0);
  return I;
}

static inline inner_t make_inner(METAL_CONSTANT const info_t& I, const int n) {
  inner_t N;
  const helper_t<> H{I, n};
  N.n = n;
  N.spots = I.spots;
  N.slice = I.slice;
  N.k0 = H.k0();
  N.k1 = H.k1();
  N.sets1 = H.sets1();
  N.sets1p_size = H.sets1p().size;
  N.sets0_offset = I.sets0_offsets[n];
  N.sets1p_offset = I.sets1p_offsets[n];
  N.cs1ps_offset = I.cs1ps_offsets[n];
  N.wins_offset = I.wins_offsets[n];
  N.input = I.spaces[n+1];
  N.output = I.spaces[n];
  return N;
}

// [:sets1.size] = all_wins1 = whether the other side wins for all possible sides
// [sets1.size:] = all_wins0_next = whether we win on the next move
static inline halfsuper_t mid_wins(METAL_CONSTANT const info_t& I, const int n, const int s) {
  const helper_t<> H{I, n};
  const auto sets1 = H.sets1();
  const bool one = s < sets1.size;
  const auto ss = one ? s : s - sets1.size;
  const auto root = I.root.side_[(I.slice+n+one)&1];
  const auto sets = one ? sets1 : H.fast_sets0_next();
  return halfsuper_wins(root | side(I.empty, sets, ss), H.parity());
}

static inline uint16_t make_cs1ps(METAL_CONSTANT const info_t& I, METAL_CONSTANT const set_t* sets1p,
                                  const int n, const int index) {
  const helper_t<> H{I, n};
  const int k0 = H.k0();
  const int k1 = H.k1();
  const int s1p = index / (I.spots-k0);
  const int j = index - s1p * (I.spots-k0);
  uint16_t c = s1p;
  for (int a = 0; a < k1; a++) {
    const int s1p_a = sets1p[s1p]>>5*a&0x1f;
    if (j<s1p_a)
      c += fast_choose(s1p_a-1, a+1) - fast_choose(s1p_a, a+1);
  }
  return c;
}

static inline set0_info_t make_set0_info(METAL_CONSTANT const info_t& I, const int n, const int s0) {
  set0_info_t I0;
  const helper_t<> H{I, n};
  const auto sets0 = H.sets0();
  const int k0 = H.k0();
  const int k1 = H.k1();

  // Construct side to move
  const set_t set0 = get(sets0, s0);
  const side_t side0 = H.root0() | side(I.empty, sets0, set0);

  // Evaluate whether we win for side0 and all child sides
  I0.wins0 = halfsuper_wins(side0, H.parity()).s;

  // List empty spots after we place s0's stones
  const auto empty1 = I0.empty1;
  {
    const auto free = side_mask & ~side0;
    int next = 0;
    for (int i = 0; i < I.spots; i++)
      if (free&side_t(1)<<I.empty.empty[i])
        empty1[next++] = i;
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
  for (int i = 0; i < I.spots-k0; i++) {
    const int j = empty1[i]-i;
    I0.child_s0s[i] = choose(empty1[i], j+1);
    for (int a = 0; a < k0; a++)
      I0.child_s0s[i] += choose(set0>>5*a&0x1f, a+(a>=j)+1);
  }

  // Lookup table to convert s0 to s0p
  for (int a = 0; a < k1; a++) {
    for (int q = 0; q < I.spots-k0; q++) {
      uint16_t offset = a ? 0 : s0;
      for (int i = empty1[q]-q; i < k0; i++) {
        const int v = set0>>5*i&0x1f;
        if (v>a)
          offset += fast_choose(v-a-1, i+1) - fast_choose(v-a, i+1);
      }
      I0.offset0[a * (I.spots-k0) + q] = offset;
    }
  }
  return I0;
}

template<class Workspace> static inline void
inner(METAL_CONSTANT const inner_t& I, METAL_CONSTANT const uint16_t* cs1ps, METAL_CONSTANT const set_t* sets1p,
      METAL_CONSTANT const halfsuper_t* all_wins, METAL_DEVICE halfsupers_t* results, const Workspace workspace,
      METAL_CONSTANT const set0_info_t& I0, const int s1p) {
  const auto set1p = sets1p[s1p];
  const auto input = slice(workspace, I.input);
  const auto output = slice(workspace, I.output);
  const auto all_wins1 = all_wins;  // all_wins1 is a prefix of all_wins
  const auto all_wins0_next = all_wins + I.sets1.size;

  // Learn that these are constants
  const int n = I.n;
  const int spots = I.spots;
  const int slice = I.slice;
  const int k0 = I.k0;
  const int k1 = I.k1;

  // Convert indices
  uint32_t filled1p = 0;
  uint16_t s1 = 0;
  uint16_t s0p = 0;
  for (int i = 0; i < k1; i++) {
    const int q = set1p>>5*i&0x1f;
    filled1p |= 1<<q;
    s1 += fast_choose(I0.empty1[q], i+1);  // I0.offset1[i][q];
    s0p += I0.offset0[i * (spots-k0) + q];
  }

  // Consider each move in turn
  halfsupers_t us;
  {
    us.win = us.notlose = halfsuper_t(0);
    if (slice + n == 36)
      us.notlose = ~halfsuper_t(0);
    auto unmoved = ~filled1p;
    for (int m = 0; m < spots-k0-k1; m++) {
      const auto bit = min_bit(unmoved);
      const int i = integer_log_exact(bit);
      unmoved &= ~bit;
      const int cs0 = I0.child_s0s[i];
      const halfsuper_t cwins = all_wins0_next[cs0];
      const halfsupers_t child = input(cs0, cs1ps[s1p*(spots-k0)+i]);
      us.win = halfsuper_t(us.win) | cwins | child.win;
      us.notlose = halfsuper_t(us.notlose) | cwins | child.notlose;
    }
  }

  // Account for immediate results
  const halfsuper_t wins0(I0.wins0);
  const auto wins1 = all_wins1[s1];
  const auto inplay = ~(wins0 | wins1);
  us.win = (inplay & us.win) | (wins0 & ~wins1);  // win (| ~inplay & (wins0 & ~wins1))
  us.notlose = (inplay & us.notlose) | wins0;  // not lose       (| ~inplay & (wins0 | ~wins1))

  // If we're far enough along, remember results
  if (n <= 1)
    results[n + s1p] = us;

  // Negate and apply rmax in preparation for the slice above
  halfsupers_t above;
  above.win = rmax(~halfsuper_t(us.notlose));
  above.notlose = rmax(~halfsuper_t(us.win));
  output(s1,s0p) = above;
}

END_NAMESPACE_PENTAGO
