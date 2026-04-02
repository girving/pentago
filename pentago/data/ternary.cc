// Flat array of ternary values, packed 5 per byte

#include "pentago/data/ternary.h"
#include "pentago/utility/ceil_div.h"
#include "pentago/utility/debug.h"
#include "pentago/utility/range.h"
#include "pentago/utility/sse.h"
#include "pentago/utility/threefry.h"
#include <algorithm>
#include <atomic>
#include <cassert>
namespace pentago {

using std::min;

// pow3[i] = 3^i
static const int pow3[5] = {1, 3, 9, 27, 81};

ternaries_t::ternaries_t()
    : size(0) {}

ternaries_t::ternaries_t(const uint64_t size)
    : size(size)
    , data(CHECK_CAST_INT(ceil_div(ceil_div(size, uint64_t(5)), uint64_t(8)))) {}

ternaries_t::~ternaries_t() {}

RawArray<uint8_t> ternaries_t::bytes() {
  return RawArray<uint8_t>(CHECK_CAST_INT(ceil_div(size, uint64_t(5))),
                           reinterpret_cast<uint8_t*>(data.data()));
}

RawArray<const uint8_t> ternaries_t::bytes() const {
  return RawArray<const uint8_t>(CHECK_CAST_INT(ceil_div(size, uint64_t(5))),
                                 reinterpret_cast<const uint8_t*>(data.data()));
}

int ternaries_t::operator[](const uint64_t i) const {
  assert(i < size);
  return bytes()[i / 5] / pow3[i % 5] % 3;
}

void ternaries_t::set(const uint64_t i, const int v) {
  assert(i < size && unsigned(v) < 3);
  const int byte = int(i / 5);
  const int pos = int(i % 5);
  auto& b = bytes()[byte];
  const int old = b / pow3[pos] % 3;
  b += uint8_t((v - old) * pow3[pos]);
}

void ternaries_t::atomic_set_from_zero(const uint64_t i, const int v) {
  assert(i < size && unsigned(v) < 3);
  auto* p = reinterpret_cast<std::atomic<uint8_t>*>(&bytes()[i / 5]);
  p->fetch_add(uint8_t(v * pow3[i % 5]), std::memory_order_relaxed);
}

#if PENTAGO_SSE

static uint64_t hsum_epu16(const __m256i v) {
  const __m256i s32 = _mm256_madd_epi16(v, _mm256_set1_epi16(1));
  const __m128i lo = _mm256_castsi256_si128(s32);
  const __m128i hi = _mm256_extracti128_si256(s32, 1);
  __m128i s = _mm_add_epi32(lo, hi);
  s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4e));
  s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xb1));
  return _mm_cvtsi128_si32(s);
}

// Count symbols 0 and 1 in packed bytes [data, data+n_bytes) using AVX2.
// Returns number of bytes processed (multiple of 32).
static int counts_avx2(const uint8_t* data, const int n_bytes, uint64_t& c0, uint64_t& c1) {
  const __m256i zero = _mm256_setzero_si256();
  const __m256i one = _mm256_set1_epi16(1);
  int i = 0;
  while (i + 32 <= n_bytes) {
    __m256i acc0 = zero, acc1 = zero;
    const int chunk_end = min(i + 32 * 1000, n_bytes & ~31);
    for (; i + 32 <= chunk_end; i += 32) {
      unpack_ternary_avx2(_mm256_loadu_si256((const __m256i*)(data + i)),
                          [&](const __m256i rlo, const __m256i rhi) {
        acc0 = _mm256_sub_epi16(acc0, _mm256_cmpeq_epi16(rlo, zero));
        acc0 = _mm256_sub_epi16(acc0, _mm256_cmpeq_epi16(rhi, zero));
        acc1 = _mm256_sub_epi16(acc1, _mm256_cmpeq_epi16(rlo, one));
        acc1 = _mm256_sub_epi16(acc1, _mm256_cmpeq_epi16(rhi, one));
      });
    }
    c0 += hsum_epu16(acc0);
    c1 += hsum_epu16(acc1);
  }
  return i;
}

#endif

Vector<uint64_t,3> ternaries_t::counts() const {
  const auto b = bytes();
  const int n_bytes = b.size();
  uint64_t c0 = 0, c1 = 0;
  int i = 0;

#if PENTAGO_SSE
  i = counts_avx2(b.data(), n_bytes, c0, c1);
#endif

  // Scalar tail
  for (; i < n_bytes; i++) {
    int v = b[i];
    for (int d = 0; d < 5; d++) {
      if (v % 3 == 0) c0++;
      else if (v % 3 == 1) c1++;
      v /= 3;
    }
  }

  // Correct for padding in the last byte
  const uint64_t total_in_bytes = uint64_t(n_bytes) * 5;
  const uint64_t c2 = total_in_bytes - c0 - c1;
  // Subtract counts from padding positions (always 0)
  const int padding = int(total_in_bytes - size);
  return vec(c0 - padding, c1, c2);
}

void ternaries_t::fill_random(const uint128_t key, const Vector<uint16_t,2> thresholds) {
  // Each threefry(key, counter) gives 128 bits = 8 uint16 random values.
  // Each uint16 r maps to: r < thresholds[0] → 0, r < thresholds[1] → 1, else → 2.
  // 8 symbols are packed into bytes via ternary_writer_t.
  ternary_writer_t writer(*this);
  const uint64_t n = size;
  uint64_t counter = 0;
  for (uint64_t i = 0; i < n; i += 8) {
    const uint128_t h = threefry(key, counter++);
    const int count = int(min(uint64_t(8), n - i));
    for (int j = 0; j < count; j++) {
      const uint16_t r = uint16_t(h >> (16 * j));
      writer.put(r < thresholds[0] ? 0 : r < thresholds[1] ? 1 : 2);
    }
  }
  writer.flush();
}

static void unpack5(const uint8_t byte, int out[5]) {
  int v = byte;
  for (int i = 0; i < 5; i++) { out[i] = v % 3; v /= 3; }
}

static uint8_t pack5(const int in[5]) {
  return uint8_t(in[0] + 3*(in[1] + 3*(in[2] + 3*(in[3] + 3*in[4]))));
}

ternary_reader_t::ternary_reader_t(const ternaries_t& t)
    : ptr(t.bytes().data())
    , end(ptr + t.bytes().size())
    , pos(5) {}

int ternary_reader_t::next() {
  if (pos >= 5) {
    unpack5(ptr < end ? *ptr++ : 0, buf);
    pos = 0;
  }
  return buf[pos++];
}

ternary_writer_t::ternary_writer_t(ternaries_t& t)
    : ptr(t.bytes().data())
    , pos(0) {
  for (int i = 0; i < 5; i++) buf[i] = 0;
}

void ternary_writer_t::put(const int v) {
  assert(unsigned(v) < 3);
  buf[pos++] = v;
  if (pos >= 5) {
    *ptr++ = pack5(buf);
    pos = 0;
  }
}

void ternary_writer_t::flush() {
  if (pos > 0) {
    for (int i = pos; i < 5; i++) buf[i] = 0;
    *ptr++ = pack5(buf);
    pos = 0;
  }
}

}  // namespace pentago
