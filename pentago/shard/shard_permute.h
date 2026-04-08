// L/H overlapping-window bit-level permutation for shard index shuffling
//
// Replaces the modular Feistel network with a sequence of bit-level shift-xor
// and shift-add operations on two overlapping power-of-two windows:
//   L window: [0, 2^k)        where k = floor(log2(n))
//   H window: [n-2^k, n)
// Their union covers [0, n). Composition of bijections on these windows
// produces a bijection on [0, n) with zero dropped entries.
//
// Each step applies one of:
//   RSX: x ^= x >> s   (right-shift-xor, linear over GF(2))
//   LSA: x += x << s   (left-shift-add, nonlinear via carries)
// Both are bijections on [0, 2^k).
//
// 8 ops in 4 LL-HH batches. Each batch applies RSX then LSA, so both
// windows get both linear and nonlinear mixing. Only 4 blends in SIMD.
//
// SIMD cost: ~32 instructions per register ≈ 10-15 cycles for 8 values.
// Measured: 1.8 ns/call (4.5 cycles/value at 2.5 GHz), 3.6x faster than
// the old multiply-based Feistel.
//
// Chi-squared quality varies with L/H overlap. Slices with large overlap
// (>20%) achieve chi2≈15 (random-quality). Slices with small overlap (<10%)
// have chi2≈50-200 (sufficient for ML training with batch sizes >= 64).
#pragma once

#include "pentago/utility/sse.h"
#include <cstdint>
namespace pentago {

using std::uint8_t;
using std::uint64_t;

static constexpr int PERM_STEPS = 8;

struct permute_constants_t {
  uint64_t n;                  // total entries for this slice
  uint8_t shifts[PERM_STEPS];  // shift amounts for each step
};

// Hardcoded for all 19 slices (0-18). Found by shard_permute_search.
inline constexpr permute_constants_t permute_constants[19] = {
  {256, {3,2,1,5,2,2,3,6}},                // slice 0
  {768, {6,6,4,1,3,2,6,4}},                // slice 1
  {9216, {4,6,4,8,7,6,6,3}},               // slice 2
  {73216, {14,6,13,12,9,7,10,3}},           // slice 3
  {720896, {6,13,8,10,6,15,4,1}},           // slice 4
  {4037632, {15,17,6,16,15,13,5,18}},       // slice 5
  {27040768, {19,21,17,16,23,16,11,20}},    // slice 6
  {144645120, {15,26,10,16,13,16,9,22}},    // slice 7
  {832507904, {22,23,17,24,16,19,25,21}},   // slice 8
  {3550828544, {11,24,29,23,16,21,23,1}},   // slice 9
  {15994707968, {24,27,3,24,7,29,25,30}},   // slice 10
  {60293679104, {27,30,7,27,1,28,9,31}},    // slice 11
  {231269590016, {29,33,10,34,7,29,1,31}},  // slice 12
  {708482302976, {18,7,4,31,34,26,18,2}},   // slice 13
  {2180723700736, {24,29,14,35,15,22,10,18}}, // slice 14
  {5720741273600, {4,9,23,41,10,37,32,35}},   // slice 15
  {14930328007168, {8,35,33,34,14,38,7,5}},   // slice 16
  {31389244375296, {4,42,19,41,5,39,17,37}},  // slice 17
  {65007135675648, {33,40,33,39,5,42,34,37}}, // slice 18
};

// Window pattern: LL-HH-LL-HH (batches of 2 on same window)
static constexpr int step_window[PERM_STEPS] = {0, 0, 1, 1, 0, 0, 1, 1};
// Op pattern: each batch does RSX then LSA
static constexpr int step_op[PERM_STEPS] = {0, 1, 0, 1, 0, 1, 0, 1};

// Scalar forward ops on k-bit domain
static inline uint64_t apply_rsx(uint64_t x, const int s) { return x ^ (x >> s); }
static inline uint64_t apply_lsa(uint64_t x, const int s, const uint64_t mask) {
  return (x + (x << s)) & mask;
}

// Scalar inverse ops (doubling-shift method)
// RSX inverse: (I+R^s)^{-1} = (I+R^s)(I+R^{2s})(I+R^{4s})... in GF(2)
// LSA inverse: (1+2^s)^{-1} = (1-2^s)(1+2^{2s})(1+2^{4s})... mod 2^k
static inline uint64_t invert_rsx(uint64_t x, const int s, const int k) {
  for (int s2 = s; s2 < k; s2 *= 2) x ^= x >> s2;
  return x;
}
static inline uint64_t invert_lsa(uint64_t x, const int s, const int k, const uint64_t mask) {
  x = (x - (x << s)) & mask;
  for (int s2 = 2 * s; s2 < k; s2 *= 2) x = (x + (x << s2)) & mask;
  return x;
}

static inline uint64_t apply_step(uint64_t x, const int op, const int s, const int k,
                                   const uint64_t mask) {
  return op == 0 ? apply_rsx(x, s) : apply_lsa(x, s, mask);
}
static inline uint64_t invert_step(uint64_t x, const int op, const int s, const int k,
                                    const uint64_t mask) {
  return op == 0 ? invert_rsx(x, s, k) : invert_lsa(x, s, k, mask);
}

#if PENTAGO_SSE
// 8 packed uint64 values in two __m256i registers
struct uint64x8 {
  __m256i v0, v1;  // [0..3], [4..7]
};

static inline __m256i simd_rsx(__m256i x, const int s) {
  return _mm256_xor_si256(x, _mm256_srli_epi64(x, s));
}
static inline __m256i simd_lsa(__m256i x, const int s, const __m256i mask) {
  return _mm256_and_si256(_mm256_add_epi64(x, _mm256_slli_epi64(x, s)), mask);
}
static inline __m256i simd_invert_rsx(__m256i x, const int s, const int k) {
  for (int s2 = s; s2 < k; s2 *= 2) x = _mm256_xor_si256(x, _mm256_srli_epi64(x, s2));
  return x;
}
static inline __m256i simd_invert_lsa(__m256i x, const int s, const int k, const __m256i mask) {
  x = _mm256_and_si256(_mm256_sub_epi64(x, _mm256_slli_epi64(x, s)), mask);
  for (int s2 = 2 * s; s2 < k; s2 *= 2)
    x = _mm256_and_si256(_mm256_add_epi64(x, _mm256_slli_epi64(x, s2)), mask);
  return x;
}
static inline __m256i simd_apply(__m256i x, const int op, const int s, const int k,
                                  const __m256i mask) {
  return op == 0 ? simd_rsx(x, s) : simd_lsa(x, s, mask);
}
static inline __m256i simd_invert(__m256i x, const int op, const int s, const int k,
                                   const __m256i mask) {
  return op == 0 ? simd_invert_rsx(x, s, k) : simd_invert_lsa(x, s, k, mask);
}
#endif  // PENTAGO_SSE

struct shard_permute_t {
  const uint64_t n;
  const int k;
  const uint64_t pow2k;
  const uint64_t offset;  // n - pow2k (H window start)
  const uint64_t mask;    // pow2k - 1
  const int shifts[PERM_STEPS];

  explicit shard_permute_t(const permute_constants_t c)
    : n(c.n),
      k(63 - __builtin_clzll(n)),
      pow2k(uint64_t(1) << k),
      offset(n - pow2k),
      mask(pow2k - 1),
      shifts{c.shifts[0], c.shifts[1], c.shifts[2], c.shifts[3],
             c.shifts[4], c.shifts[5], c.shifts[6], c.shifts[7]} {}

  explicit shard_permute_t(const int slice)
    : shard_permute_t(permute_constants[slice]) {}

  // Forward permutation: x in [0, n) -> y in [0, n)
  uint64_t forward(uint64_t x) const {
    #pragma GCC unroll 4
    for (int i = 0; i < PERM_STEPS; i += 2) {
      if (step_window[i] == 0) {
        if (x < pow2k) {
          x = apply_step(x, step_op[i], shifts[i], k, mask);
          x = apply_step(x, step_op[i + 1], shifts[i + 1], k, mask);
        }
      } else {
        if (x >= offset) {
          x -= offset;
          x = apply_step(x, step_op[i], shifts[i], k, mask);
          x = apply_step(x, step_op[i + 1], shifts[i + 1], k, mask);
          x += offset;
        }
      }
    }
    return x;
  }

  // Inverse permutation: y in [0, n) -> x in [0, n)
  uint64_t inverse(uint64_t x) const {
    for (int i = PERM_STEPS - 2; i >= 0; i -= 2) {
      if (step_window[i] == 0) {
        if (x < pow2k) {
          x = invert_step(x, step_op[i + 1], shifts[i + 1], k, mask);
          x = invert_step(x, step_op[i], shifts[i], k, mask);
        }
      } else {
        if (x >= offset) {
          x -= offset;
          x = invert_step(x, step_op[i + 1], shifts[i + 1], k, mask);
          x = invert_step(x, step_op[i], shifts[i], k, mask);
          x += offset;
        }
      }
    }
    return x;
  }

#if PENTAGO_SSE
  uint64x8 forward8(const uint64x8 x) const {
    const __m256i maskv = _mm256_set1_epi64x(mask);
    const __m256i pow2k_m1 = _mm256_set1_epi64x(pow2k - 1);
    const __m256i offset_v = _mm256_set1_epi64x(offset);
    const __m256i offset_m1 = _mm256_set1_epi64x(offset - 1);
    __m256i v0 = x.v0, v1 = x.v1;

    #pragma GCC unroll 4
    for (int i = 0; i < PERM_STEPS; i += 2) {
      if (step_window[i] == 0) {
        // L batch: apply 2 ops, blend for x < pow2k
        __m256i y0 = simd_apply(v0, step_op[i], shifts[i], k, maskv);
        y0 = simd_apply(y0, step_op[i + 1], shifts[i + 1], k, maskv);
        __m256i y1 = simd_apply(v1, step_op[i], shifts[i], k, maskv);
        y1 = simd_apply(y1, step_op[i + 1], shifts[i + 1], k, maskv);
        const __m256i gt0 = _mm256_cmpgt_epi64(v0, pow2k_m1);
        const __m256i gt1 = _mm256_cmpgt_epi64(v1, pow2k_m1);
        v0 = _mm256_blendv_epi8(y0, v0, gt0);
        v1 = _mm256_blendv_epi8(y1, v1, gt1);
      } else {
        // H batch: sub offset, apply 2 ops, add offset, blend for x >= offset
        const __m256i ge0 = _mm256_cmpgt_epi64(v0, offset_m1);
        const __m256i ge1 = _mm256_cmpgt_epi64(v1, offset_m1);
        __m256i t0 = _mm256_sub_epi64(v0, offset_v);
        __m256i t1 = _mm256_sub_epi64(v1, offset_v);
        t0 = simd_apply(t0, step_op[i], shifts[i], k, maskv);
        t0 = simd_apply(t0, step_op[i + 1], shifts[i + 1], k, maskv);
        t1 = simd_apply(t1, step_op[i], shifts[i], k, maskv);
        t1 = simd_apply(t1, step_op[i + 1], shifts[i + 1], k, maskv);
        t0 = _mm256_add_epi64(t0, offset_v);
        t1 = _mm256_add_epi64(t1, offset_v);
        v0 = _mm256_blendv_epi8(v0, t0, ge0);
        v1 = _mm256_blendv_epi8(v1, t1, ge1);
      }
    }
    return {v0, v1};
  }

  uint64x8 inverse8(const uint64x8 y) const {
    const __m256i maskv = _mm256_set1_epi64x(mask);
    const __m256i pow2k_m1 = _mm256_set1_epi64x(pow2k - 1);
    const __m256i offset_v = _mm256_set1_epi64x(offset);
    const __m256i offset_m1 = _mm256_set1_epi64x(offset - 1);
    __m256i v0 = y.v0, v1 = y.v1;

    #pragma GCC unroll 4
    for (int i = PERM_STEPS - 2; i >= 0; i -= 2) {
      if (step_window[i] == 0) {
        __m256i t0 = simd_invert(v0, step_op[i + 1], shifts[i + 1], k, maskv);
        t0 = simd_invert(t0, step_op[i], shifts[i], k, maskv);
        __m256i t1 = simd_invert(v1, step_op[i + 1], shifts[i + 1], k, maskv);
        t1 = simd_invert(t1, step_op[i], shifts[i], k, maskv);
        const __m256i gt0 = _mm256_cmpgt_epi64(v0, pow2k_m1);
        const __m256i gt1 = _mm256_cmpgt_epi64(v1, pow2k_m1);
        v0 = _mm256_blendv_epi8(t0, v0, gt0);
        v1 = _mm256_blendv_epi8(t1, v1, gt1);
      } else {
        const __m256i ge0 = _mm256_cmpgt_epi64(v0, offset_m1);
        const __m256i ge1 = _mm256_cmpgt_epi64(v1, offset_m1);
        __m256i t0 = _mm256_sub_epi64(v0, offset_v);
        __m256i t1 = _mm256_sub_epi64(v1, offset_v);
        t0 = simd_invert(t0, step_op[i + 1], shifts[i + 1], k, maskv);
        t0 = simd_invert(t0, step_op[i], shifts[i], k, maskv);
        t1 = simd_invert(t1, step_op[i + 1], shifts[i + 1], k, maskv);
        t1 = simd_invert(t1, step_op[i], shifts[i], k, maskv);
        t0 = _mm256_add_epi64(t0, offset_v);
        t1 = _mm256_add_epi64(t1, offset_v);
        v0 = _mm256_blendv_epi8(v0, t0, ge0);
        v1 = _mm256_blendv_epi8(v1, t1, ge1);
      }
    }
    return {v0, v1};
  }
#endif  // PENTAGO_SSE
};

}  // namespace pentago
