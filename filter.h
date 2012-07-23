// Multidimensional superscore filtering to precondition zlib compression
#pragma once

#include <pentago/superscore.h>
#include <other/core/utility/CopyConst.h>
namespace pentago {

// Most code should use these

void interleave(RawArray<Vector<super_t,2>> data);
void uninterleave(RawArray<Vector<super_t,2>> data);
Array<uint8_t> compact(Array<Vector<super_t,2>> src);
Array<Vector<super_t,2>> uncompact(Array<const uint8_t> src);
void wavelet_transform(RawArray<Vector<super_t,2>,4> data);
void wavelet_untransform(RawArray<Vector<super_t,2>,4> data);

// For inline use in loops elsewhere, here are the interleave primitives

static inline Vector<super_t,2> interleave_super(const Vector<super_t,2>& s) {
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
  #undef EXPAND
}

static inline Vector<super_t,2> uninterleave_super(const Vector<super_t,2>& s) {
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
  #undef CONTRACT
}

}
