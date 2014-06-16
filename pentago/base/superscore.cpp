// Operations on functions from rotations to scores

#include <pentago/base/superscore.h>
#include <pentago/base/score.h>
#include <pentago/utility/debug.h>
#include <geode/array/Array.h>
#include <geode/array/NdArray.h>
#include <geode/python/wrap.h>
#include <geode/random/Random.h>
#include <geode/utility/interrupts.h>
#include <cmath>
namespace pentago {

using std::abs;
using std::cout;
using std::endl;

int popcount(super_t s) {
#if PENTAGO_SSE
  union { __m128i a; uint64_t b[2]; } c[2];
  c[0].a = s.x;
  c[1].a = s.y;
  return popcount(c[0].b[0])+popcount(c[0].b[1])+popcount(c[1].b[0])+popcount(c[1].b[1]);
#else
  return popcount(s.a)+popcount(s.b)+popcount(s.c)+popcount(s.d);
#endif
}

struct superwin_info_t {
  super_t horizontal, vertical, diagonal_lo, diagonal_hi, diagonal_assist;
};

// We want to compute all possible rotations which give 5 in a row.
// To do this, we consider each pair or triple of quadrants which could give a win,
// and use the state space of the unused quadrants to store the various ways a win
// could be achieved.  We then do an or reduction over those quadrants.
//
// Cost: 4*5*32 = 640 bytes, 16+174 = 190 ops
super_t super_wins(side_t side) {
  // Load lookup table entries: 4*(1+3) = 16 ops
  const superwin_info_t* table = (const superwin_info_t*)superwin_info;
  #define LOAD(q) const superwin_info_t& i##q = table[512*q+quadrant(side,q)];
  LOAD(0) LOAD(1) LOAD(2) LOAD(3)

  // Prepare for reductions over unused quadrant rotations.  OR<i> is an all-reduce over the ith quadrant.
#if PENTAGO_SSE
  #define OR3 /* 3 ops */ \
    w.x |= w.y; \
    w.x |= _mm_shuffle_epi32(w.x,LE_MM_SHUFFLE(2,3,0,1)); \
    w.y = w.x;
  const int swap = LE_MM_SHUFFLE(1,0,3,2);
  #define OR2_HALF(x) /* 5 ops */ \
    x |= _mm_shuffle_epi32(x,swap); \
    x |= _mm_shufflelo_epi16(_mm_shufflehi_epi16(x,swap),swap);
  #define OR1_HALF(x) /* 8 ops */ \
    x |= _mm_slli_epi16(x,4); \
    x |= _mm_srli_epi16(x,4); \
    x |= _mm_slli_epi16(x,8); \
    x |= _mm_srli_epi16(x,8);
  #define OR0_HALF(x) /* 12 ops */ \
    x |= _mm_slli_epi16(x,1)&_mm_set1_epi8(0xaa); \
    x |= _mm_srli_epi16(x,1)&_mm_set1_epi8(0x55); \
    x |= _mm_slli_epi16(x,2)&_mm_set1_epi8(0xcc); \
    x |= _mm_srli_epi16(x,2)&_mm_set1_epi8(0x33);
  #define OR0 OR0_HALF(w.x) OR0_HALF(w.y) // 24 ops
  #define OR1 OR1_HALF(w.x) OR1_HALF(w.y) // 16 ops
  #define OR2 OR2_HALF(w.x) OR2_HALF(w.y) // 10 ops
#else // No SSE
  #define OR3 \
    w.a = w.b = w.c = w.d = w.a|w.b|w.c|w.d;
  #define OR2_PART(x) \
    x = (x|x>>32)&0x00000000ffffffff; x = x|x<<32; \
    x = (x|x>>16)&0x0000ffff0000ffff; x = x|x<<16;
  #define OR1_PART(x) \
    x = (x|x>>8)&0x00ff00ff00ff00ff; x = x|x<<8; \
    x = (x|x>>4)&0x0f0f0f0f0f0f0f0f; x = x|x<<4;
  #define OR0_PART(x) \
    x = (x|x>>2)&0x3333333333333333; x = x|x<<2; \
    x = (x|x>>1)&0x5555555555555555; x = x|x<<1;
  #define OR0 OR0_PART(w.a) OR0_PART(w.b) OR0_PART(w.c) OR0_PART(w.d)
  #define OR1 OR1_PART(w.a) OR1_PART(w.b) OR1_PART(w.c) OR1_PART(w.d)
  #define OR2 OR2_PART(w.a) OR2_PART(w.b) OR2_PART(w.c) OR2_PART(w.d)
#endif
  #define WAY(base,reduction) { super_t w = base; reduction; wins |= w; }

  // Consider all ways to win: 2*12+3*(10+16+24) = 174 ops
  super_t wins(0);
  WAY(i0.vertical & i1.vertical,     OR3 OR2) // Vertical between quadrant 0=(0,0) and 1=(0,1)
  WAY(i0.horizontal & i2.horizontal, OR3 OR1) // Horizontal between quadrant 0=(0,0) and 2=(1,0)
  WAY(i1.horizontal & i3.horizontal, OR0 OR2) // Horizontal between quadrant 1=(0,1) and 3=(1,1)
  WAY(i2.vertical & i3.vertical,     OR0 OR1) // Vertical between quadrant 2=(1,0) and 3=(1,1)
  WAY(i0.diagonal_lo & i2.diagonal_assist & i3.diagonal_lo, OR1) // Middle or low diagonal from quadrant 0=(0,0) to 3=(1,1)
  WAY(i0.diagonal_hi & i1.diagonal_assist & i3.diagonal_hi, OR2) // High diagonal from quadrant 0=(0,0) to 3=(1,1)
  WAY(i1.diagonal_lo & i0.diagonal_assist & i2.diagonal_lo, OR3) // Middle or low diagonal from quadrant 1=(0,1) to 2=(1,0)
  WAY(i1.diagonal_hi & i3.diagonal_assist & i2.diagonal_hi, OR0) // High diagonal from quadrant 1=(0,1) to 2=(1,0)
  return wins;
}

// Written in C++ so that we can be more exhaustive without being annoyingly slow
static void super_win_test(int steps) {
  // Determine the expected number of stones if we pick k stones with replacement
  Array<double> expected(32,uninit);
  for (int k=0;k<expected.size();k++)
    expected[k] = 36*(1-pow(1-1./36,k));

  // Test super_wins
  Ref<Random> random = new_<Random>(1740291);
  Array<int> counts = arange(37).copy();
  swap(counts[0],counts[5]);
  for (int count : counts) {
    // Determine how many stones we need to pick with replacement to get roughly count stones
    bool flip = count>18;
    int k = abs(expected-(flip?36-count:count)).argmin();
    GEODE_ASSERT(k<30);
    cout << "count "<<count<<", k "<<k<<", expected "<<(flip?36-expected[k]:expected[k])<<endl;

    for (int step=0;step<steps;step++) {
      // Generate a random side with roughly count stones
      side_t side = 0;
      for (int j=0;j<k;j++) {
        unsigned p = random->uniform<int>(0,36);
        side |= (side_t)1<<(16*(p%4)+p/4);
      }
      if (flip)
        side ^= side_mask;

      // Compare super_wins to won
      super_t wins = super_wins(side);
      quadrant_t rotated[4][4];
      for (int q=0;q<4;q++) {
        rotated[q][0] = quadrant(side,q);
        for (int i=0;i<3;i++)
          rotated[q][i+1] = rotations[rotated[q][i]][0];
      }
      for (int r0=0;r0<4;r0++) for (int r1=0;r1<4;r1++) for (int r2=0;r2<4;r2++) for (int r3=0;r3<4;r3++) {
        side_t rside = quadrants(rotated[0][r0],rotated[1][r1],rotated[2][r2],rotated[3][r3]);
        if (won(rside)!=wins(r0,r1,r2,r3))
          THROW(AssertionError,"side %lld, rside %lld, r %d %d %d %d, correct %d, incorrect %d",side,rside,r0,r1,r2,r3,won(rside),wins(r0,r1,r2,r3));
      }

      // Is the user impatient?
      if (!(step&1023))
        check_interrupts();
    }
  }
}

const Vector<int,4> single_rotations[8] = {vec(1,0,0,0),vec(-1,0,0,0),vec(0,1,0,0),vec(0,-1,0,0),vec(0,0,1,0),vec(0,0,-1,0),vec(0,0,0,1),vec(0,0,0,-1)};

super_t random_super(Random& random) {
  const uint64_t r0 = random.bits<uint64_t>(),
                 r1 = random.bits<uint64_t>(),
                 r2 = random.bits<uint64_t>(),
                 r3 = random.bits<uint64_t>();
  return super_t(r0,r1,r2,r3);
}

static NdArray<super_t> random_supers(const uint128_t key, Array<const int> shape) {
  NdArray<super_t> supers(shape,uninit);
  for (const int i : range(supers.flat.size())) {
    const auto ab = threefry(key,2*i),
               cd = threefry(key,2*i+1);
    supers.flat[i] = super_t(uint64_t(ab),uint64_t(ab>>64),
                             uint64_t(cd),uint64_t(cd>>64));
  }
  return supers;
}

static void super_rmax_test(int steps) {
  Ref<Random> random = new_<Random>(1740291);
  for (int step=0;step<steps;step++) {
    // Generate a random super_t
    super_t s = random_super(random);

    // Compare rmax with manual version
    super_t rs = rmax(s);
    for (int r0=0;r0<4;r0++) for (int r1=0;r1<4;r1++) for (int r2=0;r2<4;r2++) for (int r3=0;r3<4;r3++) {
      bool o = false;
      for (auto r : single_rotations)
        o |= s(vec(r0,r1,r2,r3)+r);
      GEODE_ASSERT(rs(r0,r1,r2,r3)==o);
    }

    // Is the user impatient?
    if (!(step&1023))
      check_interrupts();
  }
}

static void super_bool_test() {
  GEODE_ASSERT(!super_t(0));
  for (int i0=0;i0<4;i0++) for (int i1=0;i1<4;i1++) for (int i2=0;i2<4;i2++) for (int i3=0;i3<4;i3++) {
    GEODE_ASSERT(super_t::singleton(i0,i1,i2,i3));
    for (int j0=0;j0<4;j0++) for (int j1=0;j1<4;j1++) for (int j2=0;j2<4;j2++) for (int j3=0;j3<4;j3++)
      GEODE_ASSERT(super_t::singleton(i0,i1,i2,i3)|super_t::singleton(j0,j1,j2,j3));
  }
}

ostream& operator<<(ostream& output, superinfo_t s) {
  for (int r3=0;r3<4;r3++) {
    for (int r1=0;r1<4;r1++) {
      output << (r1||r3?' ':'[');
      for (int r2=0;r2<4;r2++) {
        if (r2) output << ' ';
        for (int r0=0;r0<4;r0++)
          output << char(s.known(r0,r1,r2,r3) ? '0'+s.wins(r0,r1,r2,r3) : '_');
      }
      output << (r1==3&&r3==3?']':'\n');
      if (r1==3) output << '\n';
    }
  }
  return output;
}

ostream& operator<<(ostream& output, super_t s) {
  superinfo_t i;
  i.known = ~super_t(0);
  i.wins = s;
  return output<<i;
}

uint8_t first(super_t s) {
  for (int r=0;r<256;r++)
    if (s(r))
      return r;
  THROW(ValueError,"zero passed to super_t first");
}

static NdArray<uint64_t> super_wins_py(NdArray<const board_t> sides) {
  Array<int> shape = sides.shape.copy();
  shape.append(4);
  NdArray<uint64_t> wins(shape,uninit);
  super_t* w = (super_t*)wins.flat.data();
  for (int i=0;i<sides.flat.size();i++) {
    GEODE_ASSERT(!(sides.flat[i]&~side_mask));
    w[i] = super_wins(sides.flat[i]);
  }
  return wins;
}

static uint64_t super_popcount(NdArray<const super_t> data) {
  uint64_t sum = 0;
  for (auto& s : data.flat)
    sum += popcount(s);
  return sum;
}

static NdArray<int> super_popcounts(NdArray<const super_t> data) {
  NdArray<int> counts(data.shape,uninit);
  for (int i=0;i<data.flat.size();i++)
    counts.flat[i] = popcount(data.flat[i]);
  return counts;
}

}
using namespace pentago;
using namespace geode::python;

void wrap_superscore() {
  GEODE_FUNCTION_2(super_wins,super_wins_py)
  GEODE_FUNCTION(super_win_test)
  GEODE_FUNCTION(super_rmax_test)
  GEODE_FUNCTION(super_bool_test)
  GEODE_FUNCTION(super_popcount)
  GEODE_FUNCTION(super_popcounts)
  GEODE_FUNCTION(random_supers)
}
