// Half of a super_t, by parity

#include "pentago/mid/halfsuper.h"
#include "pentago/data/filter.h" // For interleave and uninterleave
#include "pentago/utility/array.h"
namespace pentago {

// Mask of even parity bits in a super_t, for use in split and merge
static const uint64_t evens0 = 0xf0f00f0ff0f00f0f;
static const super_t evens(evens0,~evens0,evens0,~evens0);

// This is called outside the inner loop (about sqrt of the time in midsolve), so
// it does not need to be optimally fast.  Like merge below, it should only be roughly
// a factor of two slower.
Vector<halfsuper_t,2> split(const super_t s) {
  Vector<super_t,2> v(s&evens,s&~evens);
  v = uninterleave_super(v);
#if PENTAGO_SSE
  return vec(halfsuper_t(v[0].x | v[1].y),
             halfsuper_t(v[0].y | v[1].x));
#else
  return vec(halfsuper_t(v[0].a | v[1].c,
                         v[0].b | v[1].d),
             halfsuper_t(v[0].c | v[1].a,
                         v[0].d | v[1].b));
#endif
}

// This is called far outside the inner loop, so it does not need to be optimally fast.
// It's only a factor of two off, and reusing interleave_super is very convenient.
super_t merge(const halfsuper_t even, const halfsuper_t odd) {
  // Do two interleavings, (even,odd) and (odd,even).
  Vector<super_t,2> v;
#if PENTAGO_SSE
  v[0].x = v[1].y = even.x;
  v[0].y = v[1].x =  odd.x;
#else
  v[0].a = v[1].c = even.a;
  v[0].b = v[1].d = even.b;
  v[0].c = v[1].a =  odd.a;
  v[0].d = v[1].b =  odd.b;
#endif
  v = interleave_super(v);
  // Pick out the correct result bits
  return (v[0]&evens) | (v[1]&~evens);
}

// Clang doesn't seem smart enough to do branch free conditional swap
static inline void swap_if(uint16_t& q0, uint16_t& q1, const bool condition) {
  const uint16_t x = condition ? q0 ^ q1 : 0;
  q0 ^= x;
  q1 ^= x;
}

// One half of super_wins.  Uses only win_contributions and rotations, not superwin_info.
halfsuper_t halfsuper_wins(const side_t side, const bool parity) {
  // Grab win contributions for all rotated quadrants
  uint64_t contrib[4][4];  // Indexed by quadrant, rotation
  #pragma clang loop unroll(full)
  for (int q = 0; q < 4; q++) {
    auto q0 = quadrant(side, q);
    auto q1 = rotations[q0][0];
    auto q2 = rotations[q1][0];
    auto q3 = rotations[q0][1];
    swap_if(q0, q1, parity && !q);
    swap_if(q2, q3, parity && !q);
    contrib[q][0] = win_contributions[q][q0];
    contrib[q][1] = win_contributions[q][q1];
    contrib[q][2] = win_contributions[q][q2];
    contrib[q][3] = win_contributions[q][q3];
  }

  // Fill in the table
  halfsuper_t wins = 0;
  #pragma clang loop unroll(full)
  for (int r = 0; r < 128; r++) {
    // See won in score.h for details
    const int r1 = r >> 1 & 3;
    const int r2 = r >> 3 & 3;
    const int r3 = r >> 5;
    const int r0 = ((r & 1) << 1) + ((r1 + r2 + r3) & 1);
    const auto c = contrib[0][r0] + contrib[1][r1] + contrib[2][r2] + contrib[3][r3];
    if (c&(c>>1)&0x55 || c&(0xaaaaaaaaaaaaaaaa<<8))
      wins |= halfsuper_t::singleton(r);
  }
  return wins;
}

int popcount(halfsuper_t h) {
#if PENTAGO_SSE
  union { __m128i a; uint64_t b[2]; } c;
  c.a = h.x;
  return popcount(c.b[0]) + popcount(c.b[1]);
#else
  return popcount(h.a) + popcount(h.b);
#endif
}

}
