// Flat array of ternary values, packed 5 per byte (3^5 = 243 < 256)
#pragma once

#include "pentago/utility/array.h"
#include "pentago/utility/sse.h"
#include "pentago/utility/uint128.h"
#include "pentago/utility/vector.h"
namespace pentago {

struct ternaries_t {
  const uint64_t size;         // number of ternary values
  const Array<uint64_t> data;  // ceil(size/5) packed bytes, stored as uint64_t for alignment

  ternaries_t();
  explicit ternaries_t(const uint64_t size);
  ~ternaries_t();

  int operator[](const uint64_t i) const;
  void set(const uint64_t i, const int v);

  // Atomically set a ternary value in a zero-initialized slot.
  // Precondition: the slot at position i must be zero.
  // Safe for concurrent writes to different positions (and even the same byte).
  void atomic_set_from_zero(const uint64_t i, const int v);

  // Count occurrences of each symbol
  Vector<uint64_t,3> counts() const;

  // Fill with random ternary values using threefry.
  // Thresholds are uint16 cumulative: r < t[0] → 0, r < t[1] → 1, else → 2.
  void fill_random(const uint128_t key, const Vector<uint16_t,2> thresholds);

  // Raw byte view (length = ceil(size/5), backed by uint64_t array for alignment)
  RawArray<uint8_t> bytes();
  RawArray<const uint8_t> bytes() const;
};

// Sequential reader: unpacks 5 values at a time, avoiding per-element division
struct ternary_reader_t {
  const uint8_t* ptr;
  const uint8_t* end;
  int buf[5];
  int pos;

  explicit ternary_reader_t(const ternaries_t& t);
  int next();
};

// Sequential writer: packs 5 values at a time
struct ternary_writer_t {
  uint8_t* ptr;
  int buf[5];
  int pos;

  explicit ternary_writer_t(ternaries_t& t);
  void put(const int v);
  void flush();
};

#if PENTAGO_SSE

// AVX2: unpack 32 ternary-packed bytes into 5 rounds of 32 digits each.
// Calls f(lo_remainders, hi_remainders) for each of the 5 digit positions,
// where lo/hi are __m256i of 16x uint16 remainders in {0, 1, 2}.
template<class F>
static inline void unpack_ternary_avx2(const __m256i raw, F&& f) {
  const __m256i rcp3 = _mm256_set1_epi16(0x5556);
  const __m256i three = _mm256_set1_epi16(3);
  __m256i lo = _mm256_unpacklo_epi8(raw, _mm256_setzero_si256());
  __m256i hi = _mm256_unpackhi_epi8(raw, _mm256_setzero_si256());
  for (int d = 0; d < 5; d++) {
    const __m256i qlo = _mm256_mulhi_epu16(lo, rcp3);
    const __m256i qhi = _mm256_mulhi_epu16(hi, rcp3);
    f(_mm256_sub_epi16(lo, _mm256_mullo_epi16(qlo, three)),
      _mm256_sub_epi16(hi, _mm256_mullo_epi16(qhi, three)));
    lo = qlo;
    hi = qhi;
  }
}

#endif  // PENTAGO_SSE

}  // namespace pentago
