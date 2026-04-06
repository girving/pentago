// Modular Feistel permutation for shard index shuffling
//
// Replaces random_permute (FE2 + threefry + cycle walking) with a 3-round
// modular Feistel network on a factored domain a*b ≈ n. Each slice's n is
// a game constant; we pick a,b near sqrt(n) with a*b <= n and identity-map
// the (at most ~2700) entries in [a*b, n).
//
// Round function uses 32x32->64 multiplies (murmurhash3 constants), so scalar
// and AVX2 produce identical results. Barrett reduction avoids division.
#pragma once

#include "pentago/utility/sse.h"
#include <cmath>
#include <cstdint>
namespace pentago {

using std::swap;
using std::uint32_t;
using std::uint64_t;

struct permute_constants_t {
  uint64_t n;      // total entries for this slice
  uint32_t a;      // left half size (near sqrt(n))
};

// Hardcoded for all 19 slices (0-18). Verified by shard_permute_test.
// ratio = n/(a*b), always < 1+3e-6.
inline constexpr permute_constants_t permute_constants[19] = {
  {256, 16},              // slice 0: b=16, dropped=0, ratio=1
  {768, 24},              // slice 1: b=32, dropped=0, ratio=1
  {9216, 96},             // slice 2: b=96, dropped=0, ratio=1
  {73216, 256},           // slice 3: b=286, dropped=0, ratio=1
  {720896, 822},          // slice 4: b=877, dropped=2, ratio=1+2.8e-6
  {4037632, 1988},        // slice 5: b=2031, dropped=4, ratio=1+9.9e-7
  {27040768, 4881},       // slice 6: b=5540, dropped=28, ratio=1+1.0e-6
  {144645120, 11829},     // slice 7: b=12228, dropped=108, ratio=1+7.5e-7
  {832507904, 28492},     // slice 8: b=29219, dropped=156, ratio=1+1.9e-7
  {3550828544, 59446},    // slice 9: b=59732, dropped=72, ratio=1+2.0e-8
  {15994707968, 126017},  // slice 10: b=126925, dropped=243, ratio=1+1.5e-8
  {60293679104, 245172},  // slice 11: b=245924, dropped=176, ratio=1+2.9e-9
  {231269590016, 480734}, // slice 12: b=481076, dropped=232, ratio=1+1.0e-9
  {708482302976, 841320}, // slice 13: b=842108, dropped=416, ratio=1+5.9e-10
  {2180723700736, 1476088},  // slice 14: b=1477367, dropped=440, ratio=1+2.0e-10
  {5720741273600, 2389750},  // slice 15: b=2393866, dropped=100, ratio=1+1.7e-11
  {14930328007168, 3862621}, // slice 16: b=3865336, dropped=1512, ratio=1+1.0e-10
  {31389244375296, 5602411}, // slice 17: b=5602810, dropped=386, ratio=1+1.2e-11
  {65007135675648, 8057423}, // slice 18: b=8067981, dropped=2685, ratio=1+4.1e-11
};

// Murmurhash3 32-bit mix constants
static constexpr uint64_t C1 = 0xcc9e2d51;
static constexpr uint64_t C2 = 0x1b873593;

// Round keys (arbitrary, fixed)
static constexpr uint32_t round_keys[3] = {0x9e3779b9, 0x517cc1b7, 0x6a09e667};

// Round function: 32-bit input -> 32-bit output via multiply-xorshift.
// Uses only 32x32->64 multiplies for AVX2 compatibility.
static inline uint32_t round_fn(const uint32_t x, const uint32_t key) {
  uint64_t h = uint64_t(x ^ key) * C1;
  h ^= h >> 17;
  h = uint32_t(h) * C2;
  return uint32_t(h >> 16);
}

// Barrett modular reduction: x mod m, where x < 2^32 and m < 2^24.
// inv_m = floor(2^32 / m).
static inline uint32_t barrett_mod(const uint32_t x, const uint32_t m, const uint32_t inv_m) {
  const uint32_t q = uint32_t((uint64_t(x) * inv_m) >> 32);
  const uint32_t r = x - q * m;
  return r >= m ? r - m : r;
}

// Barrett inverse: floor(2^32 / m)
static constexpr uint32_t barrett_inv(const uint32_t m) {
  return uint32_t(uint64_t(0x100000000ULL) / m);
}

#if PENTAGO_SSE
// Pack two sets of 4 x 32-bit results (in low 32 of each 64-bit lane) into 8 x 32-bit
static inline __m256i pack32(const __m256i even, const __m256i odd) {
  const __m256i lo32 = _mm256_set1_epi64x(0xFFFFFFFF);
  return _mm256_or_si256(_mm256_and_si256(even, lo32), _mm256_slli_epi64(odd, 32));
}

// 8-wide round function on packed 32-bit elements, returning packed 32-bit results.
// Matches scalar: h = (x^key)*C1; h ^= h>>17; h = uint32(h)*C2; return uint32(h>>16)
// _mm256_mul_epu32 multiplies low 32 bits of each 64-bit lane, giving a 64-bit result.
// round_fn needs 64-bit intermediates (xor-shift on the full product), so even/odd
// elements stay wide through the pipeline and are packed at the end.
static inline __m256i simd_round(const __m256i val, const uint32_t key) {
  const __m256i keyv = _mm256_set1_epi32(key);
  const __m256i c1v = _mm256_set1_epi32(C1);
  const __m256i c2v = _mm256_set1_epi32(C2);
  const __m256i lo32 = _mm256_set1_epi64x(0xFFFFFFFF);
  const __m256i t = _mm256_xor_si256(val, keyv);
  // Even elements [0,2,4,6]: full 64-bit pipeline
  __m256i he = _mm256_mul_epu32(t, c1v);
  he = _mm256_xor_si256(he, _mm256_srli_epi64(he, 17));
  he = _mm256_mul_epu32(_mm256_and_si256(he, lo32), c2v);
  he = _mm256_srli_epi64(he, 16);
  // Odd elements [1,3,5,7]: shift into even position, same pipeline
  const __m256i t_odd = _mm256_srli_epi64(t, 32);
  __m256i ho = _mm256_mul_epu32(t_odd, c1v);
  ho = _mm256_xor_si256(ho, _mm256_srli_epi64(ho, 17));
  ho = _mm256_mul_epu32(_mm256_and_si256(ho, lo32), c2v);
  ho = _mm256_srli_epi64(ho, 16);
  return pack32(he, ho);
}

// 8-wide Barrett: q = mulhi(val, inv_m); r = val - q*m; if r >= m: r -= m
static inline __m256i simd_barrett(const __m256i val, const __m256i m, const __m256i inv_m) {
  // mulhi even/odd
  const __m256i qe = _mm256_srli_epi64(_mm256_mul_epu32(val, inv_m), 32);
  const __m256i qo = _mm256_srli_epi64(
      _mm256_mul_epu32(_mm256_srli_epi64(val, 32), _mm256_srli_epi64(inv_m, 32)), 32);
  const __m256i q = pack32(qe, qo);
  // mullo even/odd for q * m
  const __m256i qme = _mm256_mul_epu32(q, m);
  const __m256i qmo = _mm256_mul_epu32(_mm256_srli_epi64(q, 32), _mm256_srli_epi64(m, 32));
  const __m256i qm = pack32(qme, qmo);
  // r = val - q * m
  __m256i r = _mm256_sub_epi32(val, qm);
  // if r >= m: r -= m (values < 2^24, so signed cmpgt is safe)
  const __m256i ge = _mm256_or_si256(_mm256_cmpeq_epi32(r, m), _mm256_cmpgt_epi32(r, m));
  return _mm256_sub_epi32(r, _mm256_and_si256(ge, m));
}

// 8-wide modular add: (val + h) mod m, where val < m and h < m
static inline __m256i simd_add_mod(const __m256i val, const __m256i h, const __m256i m) {
  __m256i sum = _mm256_add_epi32(val, h);
  const __m256i ge = _mm256_or_si256(_mm256_cmpeq_epi32(sum, m), _mm256_cmpgt_epi32(sum, m));
  return _mm256_sub_epi32(sum, _mm256_and_si256(ge, m));
}

// 8-wide modular subtract: (val - h) mod m, where val < m and h < m
static inline __m256i simd_sub_mod(const __m256i val, const __m256i h, const __m256i m) {
  const __m256i lt = _mm256_cmpgt_epi32(h, val);
  return _mm256_add_epi32(_mm256_sub_epi32(val, h), _mm256_and_si256(lt, m));
}

// 8 packed uint64 values in two __m256i registers
struct uint64x8 {
  __m256i v0, v1;  // [0..3], [4..7]
};
#endif  // PENTAGO_SSE

struct shard_permute_t {
  const uint32_t a, b;
  const uint32_t inv_a, inv_b;
  const uint64_t ab;  // a * b
  const uint64_t n;
  const double inv_b_d;  // 1.0 / b for SIMD decomposition

  explicit shard_permute_t(const int slice)
    : a(permute_constants[slice].a),
      b(uint32_t(permute_constants[slice].n / a)),
      inv_a(barrett_inv(a)),
      inv_b(barrett_inv(b)),
      ab(uint64_t(a) * b),
      n(permute_constants[slice].n),
      inv_b_d(nextafter(1.0 / double(b), 0.0)) {}

  // Forward permutation: x in [0, n) -> y in [0, n)
  uint64_t forward(const uint64_t x) const {
    if (x >= ab) return x;  // identity for dropped entries

    // Decompose: l in [0, a), r in [0, b)
    uint32_t l = uint32_t(x / b);
    uint32_t r = uint32_t(x - uint64_t(l) * b);

    // 3 Feistel rounds
    #pragma GCC unroll 3
    for (int k = 0; k < 3; k++) {
      const uint32_t m = k & 1 ? b : a;
      const uint32_t inv_m = k & 1 ? inv_b : inv_a;
      const uint32_t h = barrett_mod(round_fn(r, round_keys[k]), m, inv_m);
      l = l + h;
      if (l >= m) l -= m;
      swap(l, r);
    }

    // After 3 swaps: l in [0, b), r in [0, a)
    return uint64_t(r) * b + l;
  }

  // Inverse permutation: y in [0, n) -> x in [0, n)
  uint64_t inverse(const uint64_t y) const {
    if (y >= ab) return y;  // identity for dropped entries

    // Decompose from final state: l in [0, b), r in [0, a)
    uint32_t r = uint32_t(y / b);
    uint32_t l = uint32_t(y - uint64_t(r) * b);

    // 3 inverse Feistel rounds (reverse order, subtract instead of add)
    #pragma GCC unroll 3
    for (int k = 2; k >= 0; k--) {
      swap(l, r);
      const uint32_t m = k & 1 ? b : a;
      const uint32_t inv_m = k & 1 ? inv_b : inv_a;
      const uint32_t h = barrett_mod(round_fn(r, round_keys[k]), m, inv_m);
      l = l >= h ? l - h : l + m - h;
    }

    // Now l in [0, a), r in [0, b): recompose as original
    return uint64_t(l) * b + r;
  }

#if PENTAGO_SSE
  // Process 8 forward permutations in parallel
  uint64x8 forward8(const uint64x8 x) const {
    const __m256i av = _mm256_set1_epi32(a);
    const __m256i bv = _mm256_set1_epi32(b);
    const __m256i inv_av = _mm256_set1_epi32(inv_a);
    const __m256i inv_bv = _mm256_set1_epi32(inv_b);
    const __m256d inv_bd = _mm256_set1_pd(inv_b_d);
    const __m256i bv64 = _mm256_set1_epi64x(b);

    // q = truncate(x / b) via precomputed double reciprocal (exact for x < 2^52)
    __m256i q0 = _mm256_cvttpd_epu64(_mm256_mul_pd(_mm256_cvtepu64_pd(x.v0), inv_bd));
    __m256i q1 = _mm256_cvttpd_epu64(_mm256_mul_pd(_mm256_cvtepu64_pd(x.v1), inv_bd));

    // r = x - q * b (64-bit). q is in low 32 of each 64-bit lane, so mul_epu32 works.
    // inv_b_d is biased down (via nextafter), so q is never too large, only possibly too small by 1.
    __m256i r0 = _mm256_sub_epi64(x.v0, _mm256_mul_epu32(q0, bv64));
    __m256i r1 = _mm256_sub_epi64(x.v1, _mm256_mul_epu32(q1, bv64));

    // Correct if q too small (r >= b): q++, r -= b
    const __m256i bm1 = _mm256_set1_epi64x(b - 1);
    const __m256i big0 = _mm256_cmpgt_epi64(r0, bm1);
    q0 = _mm256_sub_epi64(q0, big0);
    r0 = _mm256_sub_epi64(r0, _mm256_and_si256(big0, bv64));
    const __m256i big1 = _mm256_cmpgt_epi64(r1, bm1);
    q1 = _mm256_sub_epi64(q1, big1);
    r1 = _mm256_sub_epi64(r1, _mm256_and_si256(big1, bv64));

    // Pack q (=l) and r from 2 x (4 x uint64) to 8 x uint32
    const __m256i perm = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
    __m256i L = _mm256_inserti128_si256(
        _mm256_permutevar8x32_epi32(q0, perm),
        _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(q1, perm)), 1);
    __m256i R = _mm256_inserti128_si256(
        _mm256_permutevar8x32_epi32(r0, perm),
        _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(r1, perm)), 1);

    // 3 Feistel rounds
    #pragma GCC unroll 3
    for (int k = 0; k < 3; k++) {
      const __m256i mv = k & 1 ? bv : av;
      const __m256i inv_mv = k & 1 ? inv_bv : inv_av;
      const __m256i h = simd_barrett(simd_round(R, round_keys[k]), mv, inv_mv);
      L = simd_add_mod(L, h, mv);
      swap(L, R);
    }

    // Recompose: y = R * b + L, widening to 64-bit in two batches
    const __m128i L_lo = _mm256_castsi256_si128(L);
    const __m128i L_hi = _mm256_extracti128_si256(L, 1);
    const __m128i R_lo = _mm256_castsi256_si128(R);
    const __m128i R_hi = _mm256_extracti128_si256(R, 1);
    __m256i y0 = _mm256_add_epi64(_mm256_mul_epu32(_mm256_cvtepu32_epi64(R_lo), bv64),
                                   _mm256_cvtepu32_epi64(L_lo));
    __m256i y1 = _mm256_add_epi64(_mm256_mul_epu32(_mm256_cvtepu32_epi64(R_hi), bv64),
                                   _mm256_cvtepu32_epi64(L_hi));

    // Identity blend: for x >= ab, use x instead of y
    const __m256i ab_m1 = _mm256_set1_epi64x(ab - 1);
    y0 = _mm256_blendv_epi8(y0, x.v0, _mm256_cmpgt_epi64(x.v0, ab_m1));
    y1 = _mm256_blendv_epi8(y1, x.v1, _mm256_cmpgt_epi64(x.v1, ab_m1));

    return {y0, y1};
  }

  // Process 8 inverse permutations in parallel
  uint64x8 inverse8(const uint64x8 y) const {
    const __m256i av = _mm256_set1_epi32(a);
    const __m256i bv = _mm256_set1_epi32(b);
    const __m256i inv_av = _mm256_set1_epi32(inv_a);
    const __m256i inv_bv = _mm256_set1_epi32(inv_b);
    const __m256d inv_bd = _mm256_set1_pd(inv_b_d);
    const __m256i bv64 = _mm256_set1_epi64x(b);

    // divmod by b: quotient -> R, remainder -> L (inverse of forward's layout)
    __m256i q0 = _mm256_cvttpd_epu64(_mm256_mul_pd(_mm256_cvtepu64_pd(y.v0), inv_bd));
    __m256i q1 = _mm256_cvttpd_epu64(_mm256_mul_pd(_mm256_cvtepu64_pd(y.v1), inv_bd));

    __m256i rem0 = _mm256_sub_epi64(y.v0, _mm256_mul_epu32(q0, bv64));
    __m256i rem1 = _mm256_sub_epi64(y.v1, _mm256_mul_epu32(q1, bv64));

    const __m256i bm1 = _mm256_set1_epi64x(b - 1);
    const __m256i big0 = _mm256_cmpgt_epi64(rem0, bm1);
    q0 = _mm256_sub_epi64(q0, big0);
    rem0 = _mm256_sub_epi64(rem0, _mm256_and_si256(big0, bv64));
    const __m256i big1 = _mm256_cmpgt_epi64(rem1, bm1);
    q1 = _mm256_sub_epi64(q1, big1);
    rem1 = _mm256_sub_epi64(rem1, _mm256_and_si256(big1, bv64));

    // Pack: L = remainder, R = quotient
    const __m256i perm = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
    __m256i L = _mm256_inserti128_si256(
        _mm256_permutevar8x32_epi32(rem0, perm),
        _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(rem1, perm)), 1);
    __m256i R = _mm256_inserti128_si256(
        _mm256_permutevar8x32_epi32(q0, perm),
        _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(q1, perm)), 1);

    // 3 inverse Feistel rounds (reverse order, subtract instead of add)
    #pragma GCC unroll 3
    for (int k = 2; k >= 0; k--) {
      swap(L, R);
      const __m256i mv = k & 1 ? bv : av;
      const __m256i inv_mv = k & 1 ? inv_bv : inv_av;
      const __m256i h = simd_barrett(simd_round(R, round_keys[k]), mv, inv_mv);
      L = simd_sub_mod(L, h, mv);
    }

    // Recompose: x = L * b + R (L in [0,a), R in [0,b))
    const __m128i L_lo = _mm256_castsi256_si128(L);
    const __m128i L_hi = _mm256_extracti128_si256(L, 1);
    const __m128i R_lo = _mm256_castsi256_si128(R);
    const __m128i R_hi = _mm256_extracti128_si256(R, 1);
    __m256i x0 = _mm256_add_epi64(_mm256_mul_epu32(_mm256_cvtepu32_epi64(L_lo), bv64),
                                   _mm256_cvtepu32_epi64(R_lo));
    __m256i x1 = _mm256_add_epi64(_mm256_mul_epu32(_mm256_cvtepu32_epi64(L_hi), bv64),
                                   _mm256_cvtepu32_epi64(R_hi));

    // Identity blend: for y >= ab, use y instead of x
    const __m256i ab_m1 = _mm256_set1_epi64x(ab - 1);
    x0 = _mm256_blendv_epi8(x0, y.v0, _mm256_cmpgt_epi64(y.v0, ab_m1));
    x1 = _mm256_blendv_epi8(x1, y.v1, _mm256_cmpgt_epi64(y.v1, ab_m1));

    return {x0, x1};
  }
#endif  // PENTAGO_SSE
};

}  // namespace pentago
