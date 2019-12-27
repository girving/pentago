// Operations on functions from rotations to scores

#include "pentago/base/superscore.h"
#include "pentago/base/score.h"
#include "pentago/utility/array.h"
#include "pentago/utility/debug.h"
#include "pentago/utility/range.h"
#include <numeric>
#ifndef __wasm__
#include "pentago/utility/random.h"
#include <cmath>
#endif  // !__wasm__
namespace pentago {

using std::min;
using std::numeric_limits;
using std::swap;

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

const Vector<int,4> single_rotations[8] = {
    vec(1,0,0,0),vec(-1,0,0,0),vec(0,1,0,0),vec(0,-1,0,0),
    vec(0,0,1,0),vec(0,0,-1,0),vec(0,0,0,1),vec(0,0,0,-1)
};

uint8_t first(super_t s) {
  for (int r=0;r<256;r++)
    if (s(r))
      return r;
  THROW(ValueError,"zero passed to super_t first");
}

#ifndef __wasm__
super_t random_super(Random& random) {
  const uint64_t r0 = random.bits<uint64_t>(),
                 r1 = random.bits<uint64_t>(),
                 r2 = random.bits<uint64_t>(),
                 r3 = random.bits<uint64_t>();
  return super_t(r0,r1,r2,r3);
}

Array<super_t> random_supers(const uint128_t key, const int size) {
  Array<super_t> supers(size, uninit);
  for (const int i : range(size)) {
    const auto ab = threefry(key,2*i),
               cd = threefry(key,2*i+1);
    supers[i] = super_t(uint64_t(ab),uint64_t(ab>>64),
                        uint64_t(cd),uint64_t(cd>>64));
  }
  return supers;
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

uint64_t super_popcount(NdArray<const super_t> data) {
  uint64_t sum = 0;
  for (auto& s : data.flat())
    sum += popcount(s);
  return sum;
}

NdArray<int> super_popcounts(NdArray<const super_t> data) {
  NdArray<int> counts(data.shape(),uninit);
  for (int i=0;i<data.flat().size();i++)
    counts.flat()[i] = popcount(data.flat()[i]);
  return counts;
}
#endif  // !__wasm__

}
