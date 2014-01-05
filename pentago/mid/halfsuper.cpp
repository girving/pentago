// Half of a super_t, by parity

#include <pentago/mid/halfsuper.h>
#include <pentago/data/filter.h> // For interleave and uninterleave
#include <geode/array/Array.h>
#include <geode/python/wrap.h>
#include <geode/random/Random.h>
namespace pentago {

using std::cout;
using std::endl;

// Visually show that halfsuper_t is the best we can do: the parity
// configuration is the limit of rmax applied to a singleton.
static void view_rmax() {
  auto s = super_t::singleton(0);
  Array<super_t> seen;
  for (int n=0;;n++) {
    if (seen.contains(s))
      break;
    cout << "n = "<<n<<'\n'<<s<<"\n\n";
    seen.append(s);
    s = rmax(s);
  }
}

// For testing purposes
static halfsuper_t slow_split(const super_t s, const bool parity) {
  halfsuper_t h = 0;
  for (int a=0;a<2;a++)
    for (int b=0;b<4;b++)
      for (int c=0;c<4;c++)
        for (int d=0;d<4;d++)
          if (s(2*a+((b+c+d+parity)&1),b,c,d))
            h |= halfsuper_t::singleton(a,b,c,d);
  return h;
}

// For testing purposes
GEODE_UNUSED static super_t slow_merge(const halfsuper_t even, const halfsuper_t odd) {
  super_t s = 0;
  for (int a=0;a<2;a++)
    for (int b=0;b<4;b++)
      for (int c=0;c<4;c++)
        for (int d=0;d<4;d++) {
          if (even(a,b,c,d)) s |= super_t::singleton(2*a+((b+c+d  )&1),b,c,d);
          if ( odd(a,b,c,d)) s |= super_t::singleton(2*a+((b+c+d+1)&1),b,c,d);
        }
  return s;
}

// Mask of even parity bits in a super_t, for use in split and merge
static const uint64_t evens0 = 0xf0f00f0ff0f00f0f;
static const super_t evens(evens0,~evens0,evens0,~evens0);

// This is called outside the inner loop (about sqrt of the time in midsolve), so
// it does not need to be optimally fast.  Like merge below, it should only be roughly
// a factor of two slower.
static Vector<halfsuper_t,2> split(const super_t s) {
  Vector<super_t,2> v(s&evens,s&~evens);
  v = uninterleave_super(v);
  return vec(halfsuper_t(v.x.x|v.y.y),
             halfsuper_t(v.x.y|v.y.x));
}

// This is called far outside the inner loop, so it does not need to be optimally fast.
// It's only a factor of two off, and reusing interleave_super is very convenient.
super_t merge(const halfsuper_t even, const halfsuper_t odd) {
  // Do two interleavings, (even,odd) and (odd,even).
  Vector<super_t,2> v;
  v.x.x = v.y.y = even.x;
  v.x.y = v.y.x =  odd.x;
  v = interleave_super(v);
  // Pick out the correct result bits
  return (v.x&evens)|(v.y&~evens);
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

static superinfo_t info(const halfsuper_t h, const bool parity) {
  superinfo_t i;
  i.known = parity ? merge(0,~halfsuper_t(0)) : merge(~halfsuper_t(0),0);
  i.wins = parity ? merge(0,h) : merge(h,0);
  return i;
}

static super_t random_super_and(Random& random, const int steps) {
  super_t s = random_super(random);
  for (int i=1;i<steps;i++)
    s &= random_super(random);
  return s;
}

static void halfsuper_test(const int steps) {
  const auto random = new_<Random>(667731);
  const bool verbose = false;

  // Test split and merge
  for (int step=0;step<steps;step++) {
    const super_t s = step<256 ? super_t::singleton(step)
                               : random_super_and(random,4);
    const auto h = split(s);
    if (verbose)
      cout << "---------------------\ns\n"<<s<<"\nh0\n"<<info(h.x,0)<<"\nh1\n"<<info(h.y,1)<<endl;

    // Test split and merge
    if (verbose)
      cout << "slow h0\n"<<info(slow_split(s,0),0)
           <<"\nslow h1\n"<<info(slow_split(s,1),1)
           <<"\nmerge(h0,h1)\n"<<merge(h.x,h.y)<<endl;
    GEODE_ASSERT(slow_split(s,0)==h.x);
    GEODE_ASSERT(slow_split(s,1)==h.y);
    GEODE_ASSERT(s==merge(h.x,h.y));

    // Test rmax.  The order is flipped since rmax reversed parity.
    if (verbose) {
      cout << "rmax(s) = "<<popcount(rmax(s))<<"\n"<<rmax(s)<<endl;
      cout << "rmax(h0) = "<<popcount(rmax(h.x))<<"\n"<<info(rmax(h.x),1)<<endl;
      cout << "rmax(h1) = "<<popcount(rmax(h.y))<<"\n"<<info(rmax(h.y),0)<<endl;
    }
    GEODE_ASSERT(merge(rmax(h.y),rmax(h.x))==rmax(s));
  }

  // Test wins
  for (int step=0;step<steps;step++) {
    const side_t side = random_side(random);
    const auto h = halfsuper_wins(side);
    GEODE_ASSERT(super_wins(side)==merge(h.x,h.y));
  }
}

}
using namespace pentago;

void wrap_halfsuper() {
  GEODE_FUNCTION(view_rmax)
  GEODE_FUNCTION(halfsuper_test)
}
