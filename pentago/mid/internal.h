// Internal routines for midengine.cc and metal equivalent
#pragma once

#include "internal_c.h"
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
  side_t root0() const { return I.root.side_[(I.spots+n)&1]; }
  side_t root1() const { return I.root.side_[(I.spots+n+1)&1]; }
  sets_t sets0() const { return make_sets(I.spots, k0()); }
  sets_t sets1() const { return make_sets(I.spots, k1()); }
  sets_t sets0p() const { return make_sets(I.spots-k1(), k0()); }
  sets_t sets1p() const { return make_sets(I.spots-k0(), k1()); }
  int sets1p_size() const { return I.sets1p_offsets[n+1] - I.sets1p_offsets[n]; }
  int wins1_size() const { return I.wins1_offsets[n+1] - I.wins1_offsets[n]; }
  int cs1ps_size() const { return sets1p_size() * (I.spots-k0()); }
};

static inline grab_t make_grab(const bool end, const int nx, const int ny, const int workspace_size) {
  grab_t g;
  g.ny = ny;
  g.lo = end ? workspace_size - nx * ny : 0;
  g.size = nx * ny;
  return g;
}

static int bottleneck(const int spots) {
  int worst = 0;
  int prev = 1;
  for (int n = 0; n <= spots; n++) {
    int next = choose(spots, n+1);
    if (next) next *= choose(n+1, (n+1)/2);
    worst = max(worst, prev + next);
    prev = next;
  }
  return worst;
}

static inline info_t make_info(const high_board_t root) {
  info_t I;
  I.root = root.s;
  I.spots = 36 - root.count();
  I.empty = make_empty(root);
  I.sets0_offsets[0] = 0;
  I.sets1p_offsets[0] = 0;
  I.cs1ps_offsets[0] = 0;
  I.wins1_offsets[0] = 0;
  const int workspace_size = bottleneck(I.spots);
  for (int n = 0; n <= I.spots; n++) {
    const helper_t<METAL_THREAD const info_t&> H{I, n};
    I.sets0_offsets[n+1] = I.sets0_offsets[n] + H.sets0().size;
    I.sets1p_offsets[n+1] = I.sets1p_offsets[n] + H.sets1p().size;
    I.cs1ps_offsets[n+1] = I.cs1ps_offsets[n] + H.cs1ps_size();
    I.wins1_offsets[n+1] = I.wins1_offsets[n] + H.sets1().size;
    I.spaces[n] = make_grab(n&1, H.sets1().size, H.sets0p().size, workspace_size);
  }
  I.spaces[I.spots+1] = make_grab(0, 0, 0, 0);
  return I;
}

static inline transposed_t make_transposed(const high_board_t root) {
  transposed_t I;
  I.root = root.s;
  I.spots = 36 - root.count();
  I.empty = make_empty(root);
  I.sets0_offsets[0] = 0;
  I.sets1_offsets[0] = 0;
  I.cs0ps_offsets[0] = 0;
  const int workspace_size = bottleneck(I.spots);
  for (int n = 0; n <= I.spots; n++) {
    const helper_t<METAL_THREAD const transposed_t&> H{I, n};
    I.spaces[n] = make_grab(n&1, H.sets0().size, H.sets1p().size, workspace_size);
    I.sets0_offsets[n+1] = I.sets0_offsets[n] + H.sets0().size;
    I.sets1_offsets[n+1] = I.sets1_offsets[n] + H.sets1().size;
    I.cs0ps_offsets[n+1] = I.cs0ps_offsets[n] + H.sets0p().size * (I.spots-n);
  }
  I.spaces[I.spots+1] = make_grab(0, 0, 0, 0);
  return I;
}

static inline inner_t make_inner(METAL_CONSTANT const info_t& I, const int n) {
  inner_t N;
  const helper_t<> H{I, n};
  N.n = n;
  N.spots = I.spots;
  N.sets1_size = H.sets1().size;
  N.sets1p_size = H.sets1p().size;
  N.sets0_offset = I.sets0_offsets[n];
  N.sets1p_offset = I.sets1p_offsets[n];
  N.cs1ps_offset = I.cs1ps_offsets[n];
  N.wins1_offset = I.wins1_offsets[n];
  N.input = I.spaces[n+1];
  N.output = I.spaces[n];
  return N;
}

static inline transposed_inner_t make_inner(METAL_CONSTANT const transposed_t& I, const int n) {
  const helper_t<METAL_CONSTANT const transposed_t&> H{I, n};
  transposed_inner_t N;
  N.spots = I.spots;
  N.n = n;
  N.grid[0] = H.sets1().size;
  N.grid[1] = H.sets0p().size;
  N.input = I.spaces[n+1];
  N.output = I.spaces[n];
  N.sets0_offset = I.sets0_offsets[n];
  N.sets1_offset = I.sets1_offsets[n];
  N.cs0ps_offset = I.cs0ps_offsets[n];
  return N;
}

// wins1 = whether the other side wins for all possible sides
static inline wins1_t mid_wins1(METAL_CONSTANT const info_t& I, const int n, const int s1) {
  const helper_t<> H{I, n};
  const auto side1 = H.root1() | side(I.empty, H.sets1(), s1);
  const auto parity = H.parity();
  wins1_t w;
  w.after = halfsuper_wins(side1, parity);
  w.before = halfsuper_wins(side1, !parity);
  return w;
}

static inline uint16_t make_cs1ps(METAL_CONSTANT const info_t& I, METAL_CONSTANT const set_t* sets1p,
                                  const int n, const int index) {
  const helper_t<> H{I, n};
  const int k0 = H.k0();
  const int k1 = H.k1();
  const int s1p = index / (I.spots-k0);
  const int j = index - s1p * (I.spots-k0);
  uint16_t c = s1p;
  WASM_NOUNROLL
  for (int a = 0; a < k1; a++) {
    const int s1p_a = sets1p[s1p]>>5*a&0x1f;
    if (j<s1p_a)
      c -= fast_choose(s1p_a-1, a);
  }
  return c;
}

static inline int pop_bit(METAL_THREAD int& mask) {
  const auto bit = min_bit(uint32_t(mask));
  const int i = integer_log_exact(bit);
  mask &= ~bit;
  return i;
}

// Map from s0p to s0p_next
static inline uint16_t make_cs0ps(const sets_t sets0p, const int s0p, const int m) {
  int next = 0;
  bool before = true;
  for (int mask = subset_mask(sets0p, s0p), i = 0; i < sets0p.k; i++) {
    const int j = pop_bit(mask);
    if (m < j-i && before) {
      next += fast_choose(m+i, i+1);
      before = false;
    }
    next += fast_choose(j, i+1+!before);
  }
  if (before)
    next += fast_choose(sets0p.k + m, sets0p.k+1);
  return next;
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
    WASM_NOUNROLL
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
      WASM_NOUNROLL
      for (int i = empty1[q]-q; i < k0; i++) {
        const int v = set0>>5*i&0x1f;
        if (v>a)
          offset -= fast_choose(v-a-1, i);
      }
      I0.offset0[a * (I.spots-k0) + q] = offset;
    }
  }
  return I0;
}

typedef struct {
  uint16_t s0, s1p;
} s0s1p;

// Turn I1, s0p into (s0,s1p)
static inline s0s1p commute(METAL_CONSTANT const set1_info_t& I1, const sets_t sets0p, const int s0p) {
  uint16_t s0 = 0, s1p = I1.s1;
  for (int mask = subset_mask(sets0p, s0p), m = 0; m < sets0p.k; m++) {
    const int j = pop_bit(mask);
    s0 += fast_choose(I1.empty0[j], m+1);
    s1p -= I1.offset1p[m * sets0p.n + j];
  }
  return s0s1p{s0, s1p};
}

static inline set1_info_t make_set1_info(METAL_CONSTANT const transposed_t& I, const int n, const int s1) {
  set1_info_t I1;
  const helper_t<METAL_CONSTANT const transposed_t&> H{I, n};
  const auto sets1 = H.sets1();
  const int k0 = H.k0();
  const int k1 = H.k1();
  const auto parity = H.parity();

  // Construct side
  I1.s1 = s1;
  const set_t set1 = get(sets1, s1);
  const side_t side1 = H.root1() | side(I.empty, sets1, set1);

  // Whether we win with both parities
  I1.wins1.after = halfsuper_wins(side1, parity);
  I1.wins1.before = halfsuper_wins(side1, !parity);

  // List empty spots after we place s1's stones
  const auto empty0 = I1.empty0;
  {
    const auto free = side_mask & ~side1;
    int next = 0;
    for (int i = 0; i < I.spots; i++)
      if (free&side_t(1)<<I.empty.empty[i])
        empty0[next++] = i;
  }

  // Lookup table to compute s0-major indices
  for (int a = 0; a < k0+1; a++) {
    for (int q = 0; q < I.spots-k1; q++) {
      uint16_t s1p = 0;
      for (int i = empty0[q]-q; i < k1; i++) {
        const int v = set1>>5*i&0x1f;
        if (v>a)
          s1p += fast_choose(v-a-1, i);
      }
      I1.offset1p[a * (I.spots-k1) + q] = s1p;
    }
  }
  return I1;
}

#ifdef __METAL_VERSION__
// iOS has an unfortunate 256 MB buffer size limit, so we need to split workspace into up to two buffers
constant int chunk_size = (uint64_t(256) << 20) / sizeof(halfsuper_s);
constant int chunk_bits = 24;
static_assert(chunk_size == 1 << chunk_bits, "");

struct workspace_t {
  METAL_DEVICE halfsuper_s* chunks[2];
};

struct io_t {
  workspace_t w;
  int offset, stride;

  METAL_DEVICE halfsuper_s& operator()(const int i, const int j) const {
    const int r = i * stride + j + offset;
    return w.chunks[r >> chunk_bits][r & (chunk_size - 1)];
  }
};

io_t slice(const workspace_t w, const grab_t g) {
  return io_t{w, g.lo, g.ny};
}

#else  // !__METAL_VERSION

typedef halfsuper_s* workspace_t;

struct io_t {
  METAL_DEVICE halfsuper_s* data;
  int stride;

  METAL_DEVICE halfsuper_s& operator()(const int i, const int j) const { return data[i * stride + j]; }
};

static inline io_t slice(METAL_DEVICE halfsuper_s* workspace, const grab_t g) {
  return io_t{workspace + g.lo, g.ny};
}
#endif  // __METAL_VERSION__

static inline void
inner(METAL_CONSTANT const inner_t& I, METAL_CONSTANT const uint16_t* cs1ps, METAL_CONSTANT const set_t* sets1p,
      METAL_CONSTANT const wins1_t* all_wins1, METAL_DEVICE halfsuper_s* results, const workspace_t workspace,
      METAL_CONSTANT const set0_info_t& I0, const int s1p, const bool aggressive) {
  const auto set1p = sets1p[s1p];
  const auto input = slice(workspace, I.input);
  const auto output = slice(workspace, I.output);

  // Learn that these are constants
  const int n = I.n;
  const int spots = I.spots;
  const bool done = spots == n;
  const int k0 = n >> 1;
  const int k1 = n - k0;

  // The aggressive input is for the player to start.  This one is for the player to move.
  const bool turn_aggressive = aggressive ^ (n & 1);

  // Convert indices
  uint32_t filled1p = 0;
  uint16_t s1 = 0;
  uint16_t s0p = 0;
  for (int i = 0; i < k1; i++) {
    const int q = set1p>>5*i&0x1f;
    filled1p |= 1<<q;
    s1 += fast_choose(I0.empty1[q], i+1);
    s0p += I0.offset0[i * (spots-k0) + q];
  }

  // Consider each move in turn
  halfsuper_t us = {0};
  if (done && !turn_aggressive)
    us = ~halfsuper_t(0);
  for (int unmoved = ~filled1p, m = 0; m < spots-n; m++) {
    const int i = pop_bit(unmoved);
    us |= input(I0.child_s0s[i], cs1ps[s1p*(spots-k0)+i]);
  }

  // Account for immediate results
  const auto wins1 = all_wins1[s1];
  const auto inplay = ~(I0.wins0 | wins1.after);
  const auto mask = turn_aggressive ? ~wins1.after : ~halfsuper_t(0);
  us = (inplay & us) | (I0.wins0 & mask);

  // If we're far enough along, remember results
  if (n <= 1)
    results[n + s1p] = us;

  // Prepare for the slice above
  output(s1,s0p) = rmax(~us) | wins1.before;
}

END_NAMESPACE_PENTAGO
