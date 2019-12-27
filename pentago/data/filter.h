// Multidimensional superscore filtering to precondition compression
//
// The results of endgame computation come in the form of myriad Vector<super_t,2>'s,
// where the two super_t's give the rotations for which black and white win.  Since
// black and white can't both win, there is significant redundancy in this representation.
// To make the redundancy easier for zlib or lzma compression to detect, we interleave
// the bits of the two super_t's together as a filtering step.  This cuts the final size
// by 10% or so; run filter-test for detailed results.
//
// The rest of this file consists of failed experiments.
#pragma once

#include "pentago/base/superscore.h"
namespace pentago {

// Most code should use these

void interleave(RawArray<Vector<super_t,2>> data);
void uninterleave(RawArray<Vector<super_t,2>> data);
#ifndef __wasm__
Array<uint8_t> compact(Array<Vector<super_t,2>> src);
Array<Vector<super_t,2>> uncompact(Array<const uint8_t> src);
void wavelet_transform(RawArray<Vector<super_t,2>,4> data);
void wavelet_untransform(RawArray<Vector<super_t,2>,4> data);
#endif  // !__wasm__

// For inline use in loops elsewhere, here are the interleave primitives

#if PENTAGO_SSE // SSE versions

static inline Vector<super_t,2> interleave_super(const Vector<super_t,2>& s) {
  const __m128i mask32 = sse_pack<uint32_t>(-1,0,-1,0);
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
  super_t s00 = EXPAND(s[0].x),
          s01 = EXPAND(s[0].y),
          s10 = EXPAND(s[1].x),
          s11 = EXPAND(s[1].y);
  return Vector<super_t,2>(super_t(s00.x|_mm_slli_epi64(s10.x,1),
                                   s00.y|_mm_slli_epi64(s10.y,1)),
                           super_t(s01.x|_mm_slli_epi64(s11.x,1),
                                   s01.y|_mm_slli_epi64(s11.y,1)));
  #undef EXPAND1
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
  return Vector<super_t,2>(super_t(CONTRACT(s[0].x,s[0].y),
                                   CONTRACT(s[1].x,s[1].y)),
                           super_t(CONTRACT(_mm_srli_epi64(s[0].x,1),_mm_srli_epi64(s[0].y,1)),
                                   CONTRACT(_mm_srli_epi64(s[1].x,1),_mm_srli_epi64(s[1].y,1))));
  #undef CONTRACT1
  #undef CONTRACT
}

#else // Non-SSE versions

static inline Vector<super_t,2> interleave_super(const Vector<super_t,2>& s) {
  #define EXPAND(a) ({ /* Expand 32 bits into 64 bits with zero bits interleaved */ \
    uint64_t _a = (a)&0x00000000ffffffff; \
    _a = (_a|_a<<16) &0x0000ffff0000ffff; \
    _a = (_a|_a<<8)  &0x00ff00ff00ff00ff; \
    _a = (_a|_a<<4)  &0x0f0f0f0f0f0f0f0f; \
    _a = (_a|_a<<2)  &0x3333333333333333; \
    _a = (_a|_a<<1)  &0x5555555555555555; \
    _a; })
  #define LO(a) EXPAND(a)
  #define HI(a) EXPAND(a>>32)
  return Vector<super_t,2>(super_t(LO(s[0].a)|LO(s[1].a)<<1,
                                   HI(s[0].a)|HI(s[1].a)<<1,
                                   LO(s[0].b)|LO(s[1].b)<<1,
                                   HI(s[0].b)|HI(s[1].b)<<1),
                           super_t(LO(s[0].c)|LO(s[1].c)<<1,
                                   HI(s[0].c)|HI(s[1].c)<<1,
                                   LO(s[0].d)|LO(s[1].d)<<1,
                                   HI(s[0].d)|HI(s[1].d)<<1));
  #undef EXPAND
  #undef HI
  #undef LO
}

static inline Vector<super_t,2> uninterleave_super(const Vector<super_t,2>& s) {
  #define CONTRACT(a) ({ /* Contact every other bit of a uint64_t into 32 bits */ \
    uint64_t _a = (a)&0x5555555555555555; \
    _a = (_a|_a>>1)  &0x3333333333333333; \
    _a = (_a|_a>>2)  &0x0f0f0f0f0f0f0f0f; \
    _a = (_a|_a>>4)  &0x00ff00ff00ff00ff; \
    _a = (_a|_a>>8)  &0x0000ffff0000ffff; \
    _a = (_a|_a>>16) &0x00000000ffffffff; \
    _a; })
  #define MERGE(lo,hi) (CONTRACT(lo)|CONTRACT(hi)<<32)
  return Vector<super_t,2>(super_t(MERGE(s[0].a,s[0].b),
                                   MERGE(s[0].c,s[0].d),
                                   MERGE(s[1].a,s[1].b),
                                   MERGE(s[1].c,s[1].d)),
                           super_t(MERGE(s[0].a>>1,s[0].b>>1),
                                   MERGE(s[0].c>>1,s[0].d>>1),
                                   MERGE(s[1].a>>1,s[1].b>>1),
                                   MERGE(s[1].c>>1,s[1].d>>1)));
  #undef CONTRACT
  #undef MERGE
}

#endif

}
