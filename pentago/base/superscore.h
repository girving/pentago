// Operations on functions from rotations to scores
//
// A super_t encodes a subset of the rotation group G = Z_4^4, or equivalently a function
// from G to {0,1}.  Typically, this means the set of rotations for which one player or another
// wins.  Since |G| = 4^4 = 256, one super_t packs into two 128 bit __m128i values, or into
// 4 uint64_t's if we lack SSE.  Packing these bits together lets us compute the values of
// 256 different boards in parallel.
//
// There are two key interesting routines in this file: super_wins and rmax.  super_wins uses
// lookup tables to map a board side (the positions of one side's stones) into the rotations
// for which that player has five in a row.  rmax simulates branching over rotation by choosing
// the best possible single move rotation separately for each of the 256 bits.  Specifically,
// if R is the set of single move rotations, we have
//
//   f : G -> {0,1}
//   rmax(f)(a) = max_{b in R} f(a+b)
//
// If you think the bit twiddling code in this file is complicated, look at base/symmetry.cpp.
#pragma once

#include <pentago/base/board.h>
#include <geode/math/popcount.h>
#include <geode/math/sse.h>
#include <geode/random/forward.h>
#include <geode/vector/Vector.h>
#include <boost/detail/endian.hpp>
namespace pentago {

// Decide whether or not to use SSE
#if !defined(__SSE__)
#define PENTAGO_SSE 0
#else
#define PENTAGO_SSE 1
#ifndef BOOST_LITTLE_ENDIAN
#error "SSE is supported only in little endian mode"
#endif
#endif

using namespace geode;
using std::ostream;
using geode::popcount;

struct zero {};

// A subset of the rotation group Z_4^4 represented as a 256 bit mask.
// A rotation by (i0,i1,i2,i3) of quadrants 0,1,2,3 corresponds to bit i0+4*(i1+4*(i2+4*i3))
struct super_t {

#if PENTAGO_SSE
  __m128i x,y; // Little endian order as asserted above.
#elif defined(BOOST_LITTLE_ENDIAN)
  uint64_t a,b,c,d; // Little endian order.  Use different names to avoid confusion
#elif defined(BOOST_BIG_ENDIAN)
  uint64_t d,c,b,a; // Respect big endianness
#endif

  super_t() = default;

#if PENTAGO_SSE

  // Zero-only constructor
  super_t(zero*) {
    x = y = _mm_set1_epi32(0);
  }

  super_t(__m128i x, __m128i y)
    : x(x), y(y) {}

  super_t(uint64_t x0, uint64_t x1, uint64_t y0, uint64_t y1)
    : x(geode::pack(x0,x1)), y(geode::pack(y0,y1)) {}

  static super_t identity() {
    return super_t(_mm_set1_epi32(1),_mm_set1_epi32(0));
  }

  explicit operator bool() const {
    return _mm_movemask_epi8(~_mm_cmpeq_epi32(x|y,_mm_setzero_si128()))!=0;
  }

  super_t operator~() const {
    return super_t(~x,~y);
  }

  super_t operator|(super_t s) const {
    return super_t(x|s.x,y|s.y);
  }

  super_t operator&(super_t s) const {
    return super_t(x&s.x,y&s.y);
  }

  super_t operator^(super_t s) const {
    return super_t(x^s.x,y^s.y);
  }

  super_t operator|=(super_t s) {
    x |= s.x;
    y |= s.y;
    return *this;
  }

  super_t operator&=(super_t s) {
    x &= s.x;
    y &= s.y;
    return *this;
  }

  super_t operator^=(super_t s) {
    x ^= s.x;
    y ^= s.y;
    return *this;
  }

  // Do not use the following functions in performance critical code

  bool operator()(uint8_t r) const {
    return _mm_movemask_epi8(_mm_slli_epi16(r&128?y:x,7-(r&7)))>>(r>>3&15)&1;
  }

  bool parity() const {
    __m128i p = x^y;
    p ^= _mm_slli_epi16(p,4);
    p ^= _mm_slli_epi16(p,2);
    p ^= _mm_slli_epi16(p,1);
    return popcount((uint16_t)_mm_movemask_epi8(p))&1;
  }

#else // !defined(PENTAGO_SSE)

  // Zero-only constructor
  super_t(zero*) {
    a = b = c = d = 0;
  }

  // Use little endian argument order unconditionally
  super_t(uint64_t a, uint64_t b, uint64_t c, uint64_t d)
#if defined(BOOST_LITTLE_ENDIAN)
    : a(a), b(b), c(c), d(d) {}
#elif defined(BOOST_BIG_ENDIAN)
    : d(d), c(c), b(b), a(a) {}
#endif

  static super_t identity() {
    return super_t(1,0,0,0);
  }

  explicit operator bool() const {
    return a||b||c||d;
  }

  super_t operator~() const {
    return super_t(~a,~b,~c,~d);
  }

  super_t operator|(super_t s) const {
    return super_t(a|s.a,b|s.b,c|s.c,d|s.d);
  }

  super_t operator&(super_t s) const {
    return super_t(a&s.a,b&s.b,c&s.c,d&s.d);
  }

  super_t operator^(super_t s) const {
    return super_t(a^s.a,b^s.b,c^s.c,d^s.d);
  }

  super_t operator|=(super_t s) {
    a |= s.a;
    b |= s.b;
    c |= s.c;
    d |= s.d;
    return *this;
  }

  super_t operator&=(super_t s) {
    a &= s.a;
    b &= s.b;
    c &= s.c;
    d &= s.d;
    return *this;
  }

  super_t operator^=(super_t s) {
    a ^= s.a;
    b ^= s.b;
    c ^= s.c;
    d ^= s.d;
    return *this;
  }

  // Do not use the following functions in performance critical code

  bool operator()(uint8_t r) const {
    const uint8_t hi = r>>6, lo = r&63;
    return (hi==0?a:hi==1?b:hi==2?c:d)>>lo&1;
  }

  bool parity() const {
    return popcount(a^b^c^d)&1;
  }

#endif // defined(PENTAGO_SSE).  SSE independent functions follow.

  bool operator==(super_t s) const {
    return !(*this^s);
  }

  bool operator!=(super_t s) const {
    return bool(*this^s);
  }

  // Do not use the following functions in performance critical code

  bool operator()(int i0,int i1,int i2,int i3) const {
    return (*this)((i0&3)+4*((i1&3)+4*((i2&3)+4*(i3&3))));
  }

  bool operator()(const Vector<int,4>& r) const {
    return (*this)(r.x,r.y,r.z,r.w);
  }

  template<class I> bool operator[](const I& i) const {
    return operator()(i);
  }

  static super_t singleton(uint8_t r) {
    const uint8_t hi = r>>6;
    const uint64_t chunk = (uint64_t)1<<(r&63);
    return super_t(hi==0?chunk:0,hi==1?chunk:0,hi==2?chunk:0,hi==3?chunk:0);
  }

  static super_t singleton(int i0,int i1,int i2,int i3) {
    return singleton((i0&3)+4*((i1&3)+4*((i2&3)+4*(i3&3))));
  }

  static super_t singleton(Vector<int,4> r) {
    return singleton(r.x,r.y,r.z,r.w);
  }
};

// Which rotations we know about, and whether each is a win or loss.
struct superinfo_t {
  super_t known; // Which rotations we know about
  super_t wins; // The set of known wins.  We maintain the invariant that !(wins&~known)

  superinfo_t() {}

  // Zero-only constructor
  superinfo_t(zero*)
    : known(0), wins(0) {}

  superinfo_t(super_t known, super_t wins)
    : known(known), wins(wins) {}

  // Check the invariant
  bool valid() {
    return !(wins&~known);
  }
};

// rmax(f)(a) = max_{b in R} f(a+b)
static inline super_t rmax(super_t f) GEODE_CONST; // Definition below

// Given a single side, compute all rotations which yield five in a row
GEODE_EXPORT super_t super_wins(side_t side) GEODE_CONST;

// Do not use in performance critical code
GEODE_EXPORT extern const Vector<int,4> single_rotations[8];

GEODE_EXPORT uint8_t first(super_t s);

GEODE_EXPORT super_t random_super(Random& random);

GEODE_EXPORT ostream& operator<<(ostream& output, super_t s);
GEODE_EXPORT ostream& operator<<(ostream& output, superinfo_t s);

GEODE_EXPORT int popcount(super_t s);

#if PENTAGO_SSE // SSE version of rmax

// Version of shuffle with arguments in expected little endian order
#define LE_MM_SHUFFLE(i0,i1,i2,i3) _MM_SHUFFLE(i3,i2,i1,i0)

// 2*29+4+14 = 76 ops
static inline super_t rmax(const super_t f) {
  const uint32_t each0 = 0x11111111,
                 each1 = 0x000f000f;
  #define SHIFT_MASK(x,shift,mask) /* 2 ops */ \
    ((shift>0?_mm_slli_epi32(x,shift):_mm_srli_epi32(x,-(shift)))&_mm_set1_epi32(mask))
  const int left = LE_MM_SHUFFLE(3,0,1,2), right = LE_MM_SHUFFLE(1,2,3,0);
  #define FIRST_THREE(x) /* 9+2*8+4 = 29 ops */ \
    (  SHIFT_MASK(x, 1,~each0)   |SHIFT_MASK(x, -3,each0)       /* Rotate quadrant 0 left */ \
     | SHIFT_MASK(x,-1,~each0>>1)|SHIFT_MASK(x,  3,each0<<3)    /* Rotate quadrant 0 right */ \
     | SHIFT_MASK(x, 4,~each1)   |SHIFT_MASK(x,-12,each1)       /* Rotate quadrant 1 left */ \
     | SHIFT_MASK(x,-4,~each1>>4)|SHIFT_MASK(x, 12,each1<<12)   /* Rotate quadrant 1 right */ \
     | _mm_shufflelo_epi16(_mm_shufflehi_epi16(x,left),left)    /* Rotate quadrant 2 left */ \
     | _mm_shufflelo_epi16(_mm_shufflehi_epi16(x,right),right)) /* Rotate quadrant 2 right */
  const int swap = LE_MM_SHUFFLE(2,3,0,1);
  const __m128i sx = _mm_shuffle_epi32(f.x,swap),
                sy = _mm_shuffle_epi32(f.y,swap);
  const __m128i low = _mm_set_epi64x(0,~(uint64_t)0);
  return super_t(FIRST_THREE(f.x),FIRST_THREE(f.y))      // Rotate the first three quadrants
       | super_t((sy&low)|(sx&~low),(sx&low)|(sy&~low))  // Rotate quadrant 3 left
       | super_t((sx&low)|(sy&~low),(sy&low)|(sx&~low)); // Rotate quadrant 3 right
  #undef FIRST_THREE
  #undef SHIFT_MASK
}

#else // Non-SSE version of rmax

static inline super_t rmax(const super_t f) {
  const uint64_t each0 = 0x1111111111111111,
                 each1 = 0x000f000f000f000f,
                 each2 = 0x000000000000ffff;
  #define SHIFT_MASK(x,shift,mask) ((shift>0?x<<(shift):x>>-(shift))&(mask))
  #define FIRST_THREE(x) \
    (  SHIFT_MASK(x,  1,~each0)    |SHIFT_MASK(x, -3,each0)       /* Rotate quadrant 0 left */ \
     | SHIFT_MASK(x, -1,~each0>>1) |SHIFT_MASK(x,  3,each0<<3)    /* Rotate quadrant 0 right */ \
     | SHIFT_MASK(x,  4,~each1)    |SHIFT_MASK(x,-12,each1)       /* Rotate quadrant 1 left */ \
     | SHIFT_MASK(x, -4,~each1>>4) |SHIFT_MASK(x, 12,each1<<12)   /* Rotate quadrant 1 right */ \
     | SHIFT_MASK(x, 16,~each2)    |SHIFT_MASK(x,-48,each2)       /* Rotate quadrant 2 left */ \
     | SHIFT_MASK(x,-16,~each2>>16)|SHIFT_MASK(x, 48,each2<<48))  /* Rotate quadrant 2 right */
  return super_t(FIRST_THREE(f.a)|f.d|f.b,
                 FIRST_THREE(f.b)|f.a|f.c,
                 FIRST_THREE(f.c)|f.b|f.d,
                 FIRST_THREE(f.d)|f.c|f.a);
}

#endif

} namespace geode {
template<> struct IsScalar<pentago::super_t> : public mpl::true_ {};
}
