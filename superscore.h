// Operations on functions from rotations to scores
#pragma once

/*
The rotations state is the abelian group G = Z_4^4.  We begin with the set {0}.  Let R be the set of
single move rotations.  Assume the set of stone placement plays is fixed, and that the player to
move's score at ply k is given by a function f_k : G -> S, S = {-1,0,1}.  Assume first that the score
is evaluated only at a fixed time n.  Let g_k be the minimax score at ply k.  We have

  g_n(a) = f_n(a)
  g_k(a) = -min_{b in R} g_{k+1}(a+b), k < n

Say n = 2.  We have

  g_0(0) = -min_{a in R} -min_{b in R} f_2(a+b)
         = max_{a in R} min_{b in R} f_2(a+b)

Do the min and max operations commute?  Well, no.  What about n = 4?

  g_0(0) = max_{a in R} min_{c in R} max_{b in R} min_{c in R} f_4(a+b+c+d)

Ah: Let F be the space of functions f : G -> S.  We have the basic operation

  rmin : F -> F
  rmin(f,a) = min_{b in R} f(a+b)

so that

  g_0 = (-rmin)^n(f_n)

We have |F| = 2^256, so an element can be stored in 4 64-bit ints (32 bytes).  Now add back the
choice of stone placement.  The game state is now the larger space A * G, with a unified
score function f : A * G -> S.  We will specify that S = {-1,0,1} means lose, continue, win;
ties are considered a win for white.  There is a move function m : A -> 2^A given the available
moves from each position (incorporating side switching).  Let g : A * G -> S be the minimax
value of all positions.  We have the recurrence

  g(u,a) = f(u,a) || -min_{v in m(u), b in R} g(v,a+b)

We want to rewrite this in terms of operations on F.  Let's see

  g(u,a) = f(u,a) || -min_{v in m(u)} (-rmin(g(v)))(a)

Given a state (u,a) in A*G, 
*/

#include "board.h"
#include <other/core/math/sse.h>
#include <other/core/random/forward.h>
#include <other/core/vector/Vector.h>
#include <other/core/utility/safe_bool.h>
#include <other/core/utility/safe_bool.h>
namespace pentago {

using namespace other;
using std::ostream;

// A subset of the rotation group Z_4^4 represented as a 256 bit mask.
// A rotation by (i0,i1,i2,i3) of quadrants 0,1,2,3 corresponds to bit i0+4*(i1+4*(i2+4*i3))
struct super_t {
  __m128i x,y; // little endian order

  struct zero{};

  super_t() {}

  // Zero-only constructor
  super_t(zero*) {
    x = y = _mm_set1_epi32(0);
  }

  super_t(__m128i x, __m128i y)
    : x(x), y(y) {}

  super_t(uint64_t x0, uint64_t x1, uint64_t y0, uint64_t y1)
    : x(other::pack(x0,x1)), y(other::pack(y0,y1)) {}

  static super_t identity() {
    return super_t(_mm_set1_epi32(1),_mm_set1_epi32(0));
  }

  operator SafeBool() const {
    return safe_bool(_mm_movemask_epi8(~_mm_cmpeq_epi32(x|y,_mm_setzero_si128()))!=0);
  }

  bool operator==(super_t s) const {
    return !(*this^s);
  }

  bool operator!=(super_t s) const {
    return *this^s;
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

  bool operator()(int i0,int i1,int i2,int i3) const {
    return (*this)((i0&3)+4*((i1&3)+4*((i2&3)+4*(i3&3))));
  }

  bool operator()(const Vector<int,4>& r) const {
    return (*this)(r.x,r.y,r.z,r.w);
  }

  static super_t singleton(uint8_t r) {
    uint8_t hi = r>>6;
    uint64_t chunk = (uint64_t)1<<(r&63);
    return super_t(other::pack(hi==0?chunk:0,hi==1?chunk:0),other::pack(hi==2?chunk:0,hi==3?chunk:0));
  }

  static super_t singleton(int i0,int i1,int i2,int i3) {
    return singleton((i0&3)+4*((i1&3)+4*((i2&3)+4*(i3&3))));
  }

  static super_t singleton(Vector<int,4> r) {
    return singleton(r.x,r.y,r.z,r.w);
  }
};

// Version of shuffle with arguments in expected little endian order
#define LE_MM_SHUFFLE(i0,i1,i2,i3) _MM_SHUFFLE(i3,i2,i1,i0)

// rmax(f)(a) = max_{b in R} f(a+b)
static inline super_t rmax(super_t f) OTHER_CONST;
static inline super_t rmax(super_t f) {
  const uint32_t each0 = 0x11111111,
                 each1 = 0x000f000f;
  #define SHIFT_MASK(x,shift,mask) ((shift>0?_mm_slli_epi32(x,shift):_mm_srli_epi32(x,-(shift)))&_mm_set1_epi32(mask))
  const int left = LE_MM_SHUFFLE(3,0,1,2), right = LE_MM_SHUFFLE(1,2,3,0);
  #define FIRST_THREE(x) \
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

// Given a single side, compute all rotations which yield five in a row
extern super_t super_wins(side_t side) OTHER_CONST;

// Do not use in performance critical code
extern const Vector<int,4> single_rotations[8];

extern super_t random_super(Random& random);

extern ostream& operator<<(ostream& output, super_t s);

}
