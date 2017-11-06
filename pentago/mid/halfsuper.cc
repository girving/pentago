// Half of a super_t, by parity

#include "pentago/mid/halfsuper.h"
#include "pentago/data/filter.h" // For interleave and uninterleave
#include "pentago/utility/array.h"
#include "pentago/utility/random.h"
#include "pentago/utility/log.h"
#if PENTAGO_SSE
namespace pentago {

// Visually show that halfsuper_t is the best we can do: the parity
// configuration is the limit of rmax applied to a singleton.
void view_rmax() {
  auto s = super_t::singleton(0);
  vector<super_t> seen;
  for (int n=0;;n++) {
    if (std::count(seen.begin(), seen.end(), s))
      break;
    slog("n = %d\n%s\n", n, s);
    seen.push_back(s);
    s = rmax(s);
  }
}

// Mask of even parity bits in a super_t, for use in split and merge
static const uint64_t evens0 = 0xf0f00f0ff0f00f0f;
static const super_t evens(evens0,~evens0,evens0,~evens0);

// This is called outside the inner loop (about sqrt of the time in midsolve), so
// it does not need to be optimally fast.  Like merge below, it should only be roughly
// a factor of two slower.
Vector<halfsuper_t,2> split(const super_t s) {
  Vector<super_t,2> v(s&evens,s&~evens);
  v = uninterleave_super(v);
  return vec(halfsuper_t(v[0].x | v[1].y),
             halfsuper_t(v[0].y | v[1].x));
}

// This is called far outside the inner loop, so it does not need to be optimally fast.
// It's only a factor of two off, and reusing interleave_super is very convenient.
super_t merge(const halfsuper_t even, const halfsuper_t odd) {
  // Do two interleavings, (even,odd) and (odd,even).
  Vector<super_t,2> v;
  v[0].x = v[1].y = even.x;
  v[0].y = v[1].x =  odd.x;
  v = interleave_super(v);
  // Pick out the correct result bits
  return (v[0]&evens) | (v[1]&~evens);
}

// One half of super_wins
Vector<halfsuper_t,2> halfsuper_wins(const side_t side) {
  return split(super_wins(side));
}

int popcount(halfsuper_t h) {
  union { __m128i a; uint64_t b[2]; } c;
  c.a = h.x;
  return popcount(c.b[0])+popcount(c.b[1]);
}

}
#endif
