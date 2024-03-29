// SSE helper routines
#pragma once

#if PENTAGO_CPP
#include <type_traits>
#ifndef __wasm__
#include <iostream>
#endif  // !__wasm__
#endif  // PENTAGO_CPP

// Decide whether or not to use SSE
#if !defined(__SSE__)
#define PENTAGO_SSE 0
#else  // defined(__SSE__)
#define PENTAGO_SSE 1

#include <xmmintrin.h>
#include <emmintrin.h>
#include <immintrin.h>
#include <smmintrin.h>

#ifdef PENTAGO_BIG_ENDIAN
#error "SSE is supported only in little endian mode"
#endif

#if PENTAGO_CPP
namespace pentago {

template<class T> struct pack_type;
template<> struct pack_type<int32_t>{typedef __m128i type;};
template<> struct pack_type<int64_t>{typedef __m128i type;};
template<> struct pack_type<uint32_t>{typedef __m128i type;};
template<> struct pack_type<uint64_t>{typedef __m128i type;};

// Same as _mm_set_ps, but without the bizarre reversed ordering
template<class T> static inline typename pack_type<T>::type sse_pack(T x0, T x1);
template<class T> static inline typename pack_type<T>::type sse_pack(T x0, T x1, T x2, T x3);

template<> inline __m128i sse_pack<int32_t>(int32_t x0, int32_t x1, int32_t x2, int32_t x3) {
  return _mm_set_epi32(x3,x2,x1,x0);
}

template<> inline __m128i sse_pack<uint32_t>(uint32_t x0, uint32_t x1, uint32_t x2, uint32_t x3) {
  return _mm_set_epi32(x3,x2,x1,x0);
}

template<> inline __m128i sse_pack<int64_t>(int64_t x0, int64_t x1) {
  return _mm_set_epi64x(x1,x0);
}

template<> inline __m128i sse_pack<uint64_t>(uint64_t x0, uint64_t x1) {
  return _mm_set_epi64x(x1,x0);
}

template<class D,class S> static inline D expand(S x);

#ifndef __wasm__
static inline std::ostream& operator<<(std::ostream& os, __m128i a) {
  int x[4];
  *(__m128i*)x = a;
  return os<<'['<<x[0]<<','<<x[1]<<','<<x[2]<<','<<x[3]<<']';
}
#endif  // !__wasm__

static inline void transpose(__m128i& i0, __m128i& i1, __m128i& i2, __m128i& i3) {
  __m128 f0 = _mm_castsi128_ps(i0),
         f1 = _mm_castsi128_ps(i1),
         f2 = _mm_castsi128_ps(i2),
         f3 = _mm_castsi128_ps(i3);
  _MM_TRANSPOSE4_PS(f0,f1,f2,f3);
  i0 = _mm_castps_si128(f0);
  i1 = _mm_castps_si128(f1);
  i2 = _mm_castps_si128(f2);
  i3 = _mm_castps_si128(f3);
}

}
#endif  // PENTAGO_CPP
#endif  // __SSE__
