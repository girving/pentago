// Multidimensional superscore filtering to precondition zlib compression

#include "filter.h"
#include "superscore.h"
#include <other/core/array/Array2d.h>
#include <other/core/array/Array4d.h>
#include <other/core/array/NdArray.h>
#include <other/core/math/sse.h>
#include <other/core/python/module.h>
namespace pentago {

using std::cout;
using std::endl;

static inline uint8_t binary(bool b0, bool b1, bool b2, bool b3, bool b4, bool b5, bool b6, bool b7) {
  return b0|b1<<1|b2<<2|b3<<3|b4<<4|b5<<5|b6<<6|b7<<7;
}

static inline int ternary(int a0, int a1, int a2, int a3) {
  return a0+3*(a1+3*(a2+3*a3));
}

static inline int ternary(int a0, int a1, int a2, int a3, int a4, int a5, int a6, int a7) {
  return a0+3*(a1+3*(a2+3*(a3+3*(a4+3*(a5+3*(a6+3*a7))))));
}

static inline int val(const Vector<super_t,2>& s, int r0, int r1, int r2, int r3) {
  return s[1](r0,r1,r2,r3)-s[0](r0,r1,r2,r3)+1;
}

static inline void set_val(Vector<super_t,2>& s, int r0, int r1, int r2, int r3, int new_) {
  int old = val(s,r0,r1,r2,r3);
  super_t bit = super_t::singleton(r0,r1,r2,r3);
  if ((old<0)!=(new_<0))
    s[0] ^= bit;
  if ((old>0)!=(new_>0))
    s[1] ^= bit;
}

static Array<Vector<int,3>> count_outer_causal_cases(Array<const Vector<super_t,2>,4> data) {
  Array<Vector<int,3>> counts(81);
  const Vector<int,4> shape = data.shape;
  for (int i0=1;i0<shape[0];i0++)
  for (int i1=1;i1<shape[1];i1++)
  for (int i2=1;i2<shape[2];i2++)
  for (int i3=1;i3<shape[3];i3++)
    for (int r0=1;r0<4;r0++)
    for (int r1=1;r1<4;r1++)
    for (int r2=1;r2<4;r2++)
    for (int r3=1;r3<4;r3++)
      counts[ternary(val(data(i0-1,i1,i2,i3),r0,r1,r2,r3),
                     val(data(i0,i1-1,i2,i3),r0,r1,r2,r3),
                     val(data(i0,i1,i2-1,i3),r0,r1,r2,r3),
                     val(data(i0,i1,i2,i3-1),r0,r1,r2,r3))][val(data(i0,i1,i2,i3),r0,r1,r2,r3)]++;
  return counts;
}

static Array<Vector<int,3>> count_causal_cases(Array<const Vector<super_t,2>,4> data) {
  Array<Vector<int,3>> counts(6561);
  const Vector<int,4> shape = data.shape;
  for (int i0=1;i0<shape[0];i0++)
  for (int i1=1;i1<shape[1];i1++)
  for (int i2=1;i2<shape[2];i2++)
  for (int i3=1;i3<shape[3];i3++)
    for (int r0=1;r0<4;r0++)
    for (int r1=1;r1<4;r1++)
    for (int r2=1;r2<4;r2++)
    for (int r3=1;r3<4;r3++)
      counts[ternary(val(data(i0-1,i1,i2,i3),r0,r1,r2,r3),
                     val(data(i0,i1-1,i2,i3),r0,r1,r2,r3),
                     val(data(i0,i1,i2-1,i3),r0,r1,r2,r3),
                     val(data(i0,i1,i2,i3-1),r0,r1,r2,r3),
                     val(data(i0,i1,i2,i3),r0-1,r1,r2,r3),
                     val(data(i0,i1,i2,i3),r0,r1-1,r2,r3),
                     val(data(i0,i1,i2,i3),r0,r1,r2-1,r3),
                     val(data(i0,i1,i2,i3),r0,r1,r2,r3-1))][val(data(i0,i1,i2,i3),r0,r1,r2,r3)]++;
  return counts;
}

static void outer_causal_filter(Array<const Vector<int,3>> table, Array<Vector<super_t,2>,4> data) {
  OTHER_ASSERT(table.size()==81);
  const Vector<int,4> shape = data.shape;
  for (int i0=shape[0]-1;i0>0;i0--)
  for (int i1=shape[1]-1;i1>0;i1--)
  for (int i2=shape[2]-1;i2>0;i2--)
  for (int i3=shape[3]-1;i3>0;i3--) {
    auto& s = data(i0,i1,i2,i3);
    for (int r0=3;r0>0;r0--) 
    for (int r1=3;r1>0;r1--) 
    for (int r2=3;r2>0;r2--) 
    for (int r3=3;r3>0;r3--)
      set_val(s,r0,r1,r2,r3,table[ternary(val(data(i0-1,i1,i2,i3),r0,r1,r2,r3),
                                          val(data(i0,i1-1,i2,i3),r0,r1,r2,r3),
                                          val(data(i0,i1,i2-1,i3),r0,r1,r2,r3),
                                          val(data(i0,i1,i2,i3-1),r0,r1,r2,r3))][val(s,r0,r1,r2,r3)]);
  }
}

static void inverse_outer_causal_filter(Array<const Vector<int,3>> table, Array<Vector<super_t,2>,4> data) {
  OTHER_ASSERT(table.size()==81);
  const Vector<int,4> shape = data.shape;
  for (int i0=1;i0<shape[0];i0++)
  for (int i1=1;i1<shape[1];i1++)
  for (int i2=1;i2<shape[2];i2++)
  for (int i3=1;i3<shape[3];i3++) {
    auto& s = data(i0,i1,i2,i3);
    for (int r0=1;r0<4;r0++)
    for (int r1=1;r1<4;r1++)
    for (int r2=1;r2<4;r2++)
    for (int r3=1;r3<4;r3++) {
      const Vector<int,3>& t = table[ternary(val(data(i0-1,i1,i2,i3),r0,r1,r2,r3),
                                             val(data(i0,i1-1,i2,i3),r0,r1,r2,r3),
                                             val(data(i0,i1,i2-1,i3),r0,r1,r2,r3),
                                             val(data(i0,i1,i2,i3-1),r0,r1,r2,r3))];
      set_val(s,r0,r1,r2,r3,t.find(val(s,r0,r1,r2,r3)));
    }
  }
}

static void arbitrary_causal_filter(Array<const Vector<int,3>> table, Array<Vector<super_t,2>,4> data) {
  OTHER_ASSERT(table.size()==6561);
  const Vector<int,4> shape = data.shape;
  for (int i0=shape[0]-1;i0>0;i0--)
  for (int i1=shape[1]-1;i1>0;i1--)
  for (int i2=shape[2]-1;i2>0;i2--)
  for (int i3=shape[3]-1;i3>0;i3--) {
    auto& s = data(i0,i1,i2,i3);
    for (int r0=3;r0>0;r0--) 
    for (int r1=3;r1>0;r1--) 
    for (int r2=3;r2>0;r2--) 
    for (int r3=3;r3>0;r3--)
      set_val(s,r0,r1,r2,r3,table[ternary(val(data(i0-1,i1,i2,i3),r0,r1,r2,r3),
                                          val(data(i0,i1-1,i2,i3),r0,r1,r2,r3),
                                          val(data(i0,i1,i2-1,i3),r0,r1,r2,r3),
                                          val(data(i0,i1,i2,i3-1),r0,r1,r2,r3),
                                          val(data(i0,i1,i2,i3),r0-1,r1,r2,r3),
                                          val(data(i0,i1,i2,i3),r0,r1-1,r2,r3),
                                          val(data(i0,i1,i2,i3),r0,r1,r2-1,r3),
                                          val(data(i0,i1,i2,i3),r0,r1,r2,r3-1))][val(s,r0,r1,r2,r3)]);
  }
}

static void inverse_arbitrary_causal_filter(Array<const Vector<int,3>> table, Array<Vector<super_t,2>,4> data) {
  OTHER_ASSERT(table.size()==6561);
  const Vector<int,4> shape = data.shape;
  for (int i0=1;i0<shape[0];i0++)
  for (int i1=1;i1<shape[1];i1++)
  for (int i2=1;i2<shape[2];i2++)
  for (int i3=1;i3<shape[3];i3++) {
    auto& s = data(i0,i1,i2,i3);
    for (int r0=1;r0<4;r0++)
    for (int r1=1;r1<4;r1++)
    for (int r2=1;r2<4;r2++)
    for (int r3=1;r3<4;r3++) {
      const Vector<int,3>& t = table[ternary(val(data(i0-1,i1,i2,i3),r0,r1,r2,r3),
                                             val(data(i0,i1-1,i2,i3),r0,r1,r2,r3),
                                             val(data(i0,i1,i2-1,i3),r0,r1,r2,r3),
                                             val(data(i0,i1,i2,i3-1),r0,r1,r2,r3),
                                             val(data(i0,i1,i2,i3),r0-1,r1,r2,r3),
                                             val(data(i0,i1,i2,i3),r0,r1-1,r2,r3),
                                             val(data(i0,i1,i2,i3),r0,r1,r2-1,r3),
                                             val(data(i0,i1,i2,i3),r0,r1,r2,r3-1))];
      set_val(s,r0,r1,r2,r3,t.find(val(s,r0,r1,r2,r3)));
    }
  }
}

static Array<int,2> count_rotation_cases(Array<const Vector<super_t,2>,4> data) {
  Array<int,2> counts(4,81);
  for (auto& s : data.flat) {
    for (int r0=1;r0<4;r0++)
    for (int r1=1;r1<4;r1++)
    for (int r2=1;r2<4;r2++) {
      counts(0,ternary(val(s,0,r0,r1,r2),val(s,1,r0,r1,r2),val(s,2,r0,r1,r2),val(s,3,r0,r1,r2)))++;
      counts(1,ternary(val(s,r0,0,r1,r2),val(s,r0,1,r1,r2),val(s,r0,2,r1,r2),val(s,r0,3,r1,r2)))++;
      counts(2,ternary(val(s,r0,r1,0,r2),val(s,r0,r1,1,r2),val(s,r0,r1,2,r2),val(s,r0,r1,3,r2)))++;
      counts(3,ternary(val(s,r0,r1,r2,0),val(s,r0,r1,r2,1),val(s,r0,r1,r2,2),val(s,r0,r1,r2,3)))++;
    }
  }
  return counts;
}

static inline Vector<super_t,2> interleave(const Vector<super_t,2>& s) {
  const __m128i mask32 = other::pack<uint32_t>(-1,0,-1,0);
  #define EXPAND1(a) ({ \
    a = (a|_mm_slli_epi64(a,16))&_mm_set1_epi64x(0x0000ffff0000ffff); \
    a = (a|_mm_slli_epi64(a, 8))&_mm_set1_epi64x(0x00ff00ff00ff00ff); \
    a = (a|_mm_slli_epi64(a, 4))&_mm_set1_epi64x(0x0f0f0f0f0f0f0f0f); \
    a = (a|_mm_slli_epi64(a, 2))&_mm_set1_epi64x(0x3333333333333333); \
    a = (a|_mm_slli_epi64(a, 1))&_mm_set1_epi64x(0x5555555555555555); })
  #define EXPAND(w) ({ \
    super_t a(_mm_shuffle_epi32(w,LE_MM_SHUFFLE(0,2,1,3))&mask32, \
              _mm_shuffle_epi32(w,LE_MM_SHUFFLE(2,0,3,1))&mask32); \
    EXPAND1(a.x); \
    EXPAND1(a.y); \
    a; })
  super_t s00 = EXPAND(s.x.x),
          s01 = EXPAND(s.x.y),
          s10 = EXPAND(s.y.x),
          s11 = EXPAND(s.y.y);
  return Vector<super_t,2>(super_t(s00.x|_mm_slli_epi64(s10.x,1),
                                   s00.y|_mm_slli_epi64(s10.y,1)),
                           super_t(s01.x|_mm_slli_epi64(s11.x,1),
                                   s01.y|_mm_slli_epi64(s11.y,1)));
}

static inline Vector<super_t,2> uninterleave(const Vector<super_t,2>& s) {
  #define CONTRACT1(w) ({ \
    __m128i a = w&_mm_set1_epi64x(0x5555555555555555); \
    a = (a|_mm_srli_epi64(a, 1))&_mm_set1_epi64x(0x3333333333333333); \
    a = (a|_mm_srli_epi64(a, 2))&_mm_set1_epi64x(0x0f0f0f0f0f0f0f0f); \
    a = (a|_mm_srli_epi64(a, 4))&_mm_set1_epi64x(0x00ff00ff00ff00ff); \
    a = (a|_mm_srli_epi64(a, 8))&_mm_set1_epi64x(0x0000ffff0000ffff); \
    a = (a|_mm_srli_epi64(a,16))&_mm_set1_epi64x(0x00000000ffffffff); \
    a; })
  #define CONTRACT(a,b) ({ \
    __m128i aa = CONTRACT1(a), \
            bb = CONTRACT1(b); \
     _mm_shuffle_epi32(aa,LE_MM_SHUFFLE(0,2,1,3)) \
    |_mm_shuffle_epi32(bb,LE_MM_SHUFFLE(1,3,0,2)); })
  return Vector<super_t,2>(super_t(CONTRACT(s.x.x,s.x.y),
                                   CONTRACT(s.y.x,s.y.y)),
                           super_t(CONTRACT(_mm_srli_epi64(s.x.x,1),_mm_srli_epi64(s.x.y,1)),
                                   CONTRACT(_mm_srli_epi64(s.y.x,1),_mm_srli_epi64(s.y.y,1))));
}
 
void interleave(RawArray<Vector<super_t,2>> data) {
  for (auto& s : data)
    s = interleave(s);
}

void uninterleave(RawArray<Vector<super_t,2>> data) {
  for (auto& s : data)
    s = uninterleave(s);
}

static void interleave(NdArray<Vector<super_t,2>> data) {
  for (auto& s : data.flat)
    s = interleave(s);
}

static void uninterleave(NdArray<Vector<super_t,2>> data) {
  for (auto& s : data.flat)
    s = uninterleave(s);
}

}
using namespace pentago;

void wrap_filter() {
  OTHER_FUNCTION(count_causal_cases)
  OTHER_FUNCTION(count_outer_causal_cases)
  OTHER_FUNCTION(count_rotation_cases)
  python::function("interleave",static_cast<void(*)(NdArray<Vector<super_t,2>>)>(interleave));
  python::function("uninterleave",static_cast<void(*)(NdArray<Vector<super_t,2>>)>(uninterleave));
  OTHER_FUNCTION(arbitrary_causal_filter)
  OTHER_FUNCTION(inverse_arbitrary_causal_filter)
  OTHER_FUNCTION(outer_causal_filter)
  OTHER_FUNCTION(inverse_outer_causal_filter)
}
