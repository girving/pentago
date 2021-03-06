// Half of a super_t, by parity

#include "halfsuper.h"
#if defined(__APPLE__) && defined(__wasm__)
#include "../base/gen/halfsuper_wins.h"
#else
#include "pentago/base/gen/halfsuper_wins.h"
#endif
#include "../data/filter.h" // For interleave and uninterleave
#include "../utility/array.h"
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
  v[0].x = v[1].y = even.s.x;
  v[0].y = v[1].x =  odd.s.x;
#else
  v[0].a = v[1].c = even.s.a;
  v[0].b = v[1].d = even.s.b;
  v[0].c = v[1].a =  odd.s.a;
  v[0].d = v[1].b =  odd.s.b;
#endif
  v = interleave_super(v);
  // Pick out the correct result bits
  return (v[0]&evens) | (v[1]&~evens);
}

int popcount(halfsuper_s h) {
#if PENTAGO_SSE
  union { __m128i a; uint64_t b[2]; } c;
  c.a = h.x;
  return popcount(c.b[0]) + popcount(c.b[1]);
#else
  return popcount(h.a) + popcount(h.b);
#endif
}

}
