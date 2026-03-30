// rANS (range Asymmetric Numeral Systems) coding of ternary arrays
//
// Format (version 2): 8 independent rANS coders with 32-bit state, concatenated.
// Uses power-of-2 total for division-free decode. 8-bit renormalization.
// Each stream's bytes are stored reversed (encoder writes LIFO, decoder reads FIFO).
// AVX2 path processes all 8 lanes in parallel.

#include "pentago/data/arithmetic.h"
#include "pentago/utility/debug.h"
#include "pentago/utility/range.h"
#include "pentago/utility/sse.h"
#include <algorithm>
#include <cassert>
#include <cstring>
namespace pentago {

using std::max;
using std::reverse;

static const int lanes = arithmetic_lanes;
static const int total_bits = 14;
static const uint32_t rans_total = 1 << total_bits;
static const uint32_t rans_lower = 1 << 23;

namespace {

struct rans_freq_t {
  uint32_t freq[3];
  uint32_t cum[3];       // {0, freq[0], freq[0]+freq[1]}
  uint32_t x_max[3];     // renormalize threshold per symbol
  uint32_t rcp[3];       // reciprocal magic (add-and-shift variant)
  uint32_t rcp_shift[3]; // shift for reciprocal
};

// Reciprocal for unsigned division by d >= 2, using the Hacker's Delight add-and-shift variant:
//   n / d == (mulhi(n, magic) + ((n - mulhi(n, magic)) >> 1)) >> shift, for n in [0, 2^31).
// For d <= 1, returns zeros (handled by fixup in the AVX2 encoder).
static void compute_rcp(const uint32_t d, uint32_t& magic, uint32_t& shift) {
  if (d <= 1) { magic = 0; shift = 0; return; }
  int l = 0;
  while ((1ull << l) < d) l++;
  magic = uint32_t(((uint64_t(1) << (32 + l)) + d - 1) / d - (uint64_t(1) << 32));
  shift = l - 1;
}

}  // namespace

static rans_freq_t make_freq(const Vector<uint64_t,3> counts) {
  constexpr uint64_t max_total = numeric_limits<uint64_t>::max() / 0x3fff;
  const uint64_t sum = counts[0] + counts[1] + counts[2];
  GEODE_ASSERT(counts[0] <= sum && counts[1] <= sum && sum <= max_total);

  uint32_t f[3];
  if (sum) {
    for (int i = 0; i < 3; i++)
      f[i] = uint32_t(counts[i] * rans_total / sum);
    for (int i = 0; i < 3; i++)
      if (!f[i]) f[i] = 1;
    const int32_t adjust = int32_t(rans_total) - int32_t(f[0] + f[1] + f[2]);
    int largest = 0;
    if (f[1] > f[largest]) largest = 1;
    if (f[2] > f[largest]) largest = 2;
    GEODE_ASSERT(int32_t(f[largest]) + adjust >= 1);
    f[largest] += adjust;
  } else {
    f[0] = f[1] = f[2] = rans_total / 3;
    f[0] += rans_total - 3 * (rans_total / 3);
  }
  const uint32_t xm = (rans_lower >> total_bits) << 8;
  rans_freq_t tab = {{f[0], f[1], f[2]},
                     {0, f[0], f[0] + f[1]},
                     {xm * f[0], xm * f[1], xm * f[2]},
                     {}, {}};
  for (int i = 0; i < 3; i++)
    compute_rcp(f[i], tab.rcp[i], tab.rcp_shift[i]);
  return tab;
}

namespace {

// Scalar encode/decode

static void rans_put(uint32_t& state, const rans_freq_t& tab, const int sym,
                     uint8_t*& cursor) {
  assert(unsigned(sym) < 3);
  while (state >= tab.x_max[sym]) {
    *cursor++ = state & 0xff;
    state >>= 8;
  }
  const uint32_t freq = tab.freq[sym];
  state = ((state / freq) << total_bits) + (state % freq) + tab.cum[sym];
}

static int rans_get(uint32_t& state, const rans_freq_t& tab,
                    const uint8_t*& ptr, const uint8_t* end) {
  const uint32_t slot = state & (rans_total - 1);
  const int sym = slot < tab.cum[1] ? 0 : slot < tab.cum[2] ? 1 : 2;
  state = tab.freq[sym] * (state >> total_bits) + slot - tab.cum[sym];
  while (state < rans_lower && ptr < end)
    state = (state << 8) | *ptr++;
  return sym;
}

static arithmetic_t concat_streams(const Vector<uint64_t,3> counts,
                                   const uint8_t* const ptrs[lanes],
                                   const int lengths[lanes]) {
  Vector<uint32_t,lanes> stream_lengths;
  uint64_t total_bytes = 0;
  for (int lane = 0; lane < lanes; lane++) {
    stream_lengths[lane] = lengths[lane];
    total_bytes += lengths[lane];
  }
  Array<uint8_t> output(CHECK_CAST_INT(total_bytes), uninit);
  int offset = 0;
  for (int lane = 0; lane < lanes; lane++) {
    memcpy(output.data() + offset, ptrs[lane], lengths[lane]);
    offset += lengths[lane];
  }
  return {2, counts, stream_lengths, output};
}

#if PENTAGO_SSE

// High 32 bits of 8x (uint32 * uint32)
static __m256i mulhi_epu32(const __m256i a, const __m256i b) {
  const __m256i lo = _mm256_mul_epu32(a, b);
  const __m256i hi = _mm256_mul_epu32(_mm256_srli_epi64(a, 32), _mm256_srli_epi64(b, 32));
  return _mm256_or_si256(_mm256_srli_epi64(lo, 32),
                         _mm256_and_si256(hi, _mm256_set1_epi64x(int64_t(0xffffffff00000000LL))));
}

// Shuffle a 3-element table by 8 indices (each 0, 1, or 2).
// table8 = {t[0], t[1], t[2], 0, t[0], t[1], t[2], 0} — duplicated for cross-lane permutevar.
static __m256i shuffle3(const __m256i table8, const __m256i idx8) {
  return _mm256_permutevar8x32_epi32(table8, idx8);
}

// Build a duplicated 3-element table for shuffle3
static __m256i make_table8(const uint32_t a, const uint32_t b, const uint32_t c) {
  return _mm256_setr_epi32(a, b, c, 0, a, b, c, 0);
}

static arithmetic_t arithmetic_encode_avx2(const ternaries_t data, const rans_freq_t& tab,
                                           const Vector<uint64_t,3> counts) {
  const uint64_t n = data.size;

  // Build per-symbol lookup tables for shuffle
  const __m256i freq_tab = make_table8(tab.freq[0], tab.freq[1], tab.freq[2]);
  const __m256i cum_tab = make_table8(tab.cum[0], tab.cum[1], tab.cum[2]);
  const __m256i xmax_tab = make_table8(tab.x_max[0], tab.x_max[1], tab.x_max[2]);
  const __m256i rcp_tab = make_table8(tab.rcp[0], tab.rcp[1], tab.rcp[2]);
  const __m256i shift_tab = make_table8(tab.rcp_shift[0], tab.rcp_shift[1], tab.rcp_shift[2]);
  const __m256i one8 = _mm256_set1_epi32(1);

  __m256i state8 = _mm256_set1_epi32(rans_lower);

  // Pre-allocate output buffers with raw cursors for zero-overhead writes.
  // Max output per lane ≈ n/8 * 1.5 bytes + 4 (state flush).
  const int buf_capacity = int(n / lanes * 2) + 64;
  Array<uint8_t> buf_storage[lanes];
  uint8_t* cursors[lanes];
  for (int lane = 0; lane < lanes; lane++) {
    buf_storage[lane] = Array<uint8_t>(buf_capacity, uninit);
    cursors[lane] = buf_storage[lane].data();
  }
  auto finish = [&]() {
    const uint8_t* ptrs[lanes];
    int lengths[lanes];
    for (int lane = 0; lane < lanes; lane++) {
      ptrs[lane] = buf_storage[lane].data();
      lengths[lane] = int(cursors[lane] - buf_storage[lane].data());
    }
    return concat_streams(counts, ptrs, lengths);
  };

  // Handle tail (symbols not covered by 160-symbol batches)
  // 160 = 32 bytes * 5 digits = 20 groups of 8 lanes
  const int64_t batch = 32 * 5;  // 160 symbols per AVX2 unpack
  const int64_t full_batches = int64_t(n) / batch * batch;
  {
    alignas(32) uint32_t st[8];
    _mm256_store_si256((__m256i*)st, state8);
    for (int64_t i = int64_t(n) - 1; i >= full_batches; i--)
      rans_put(st[i % lanes], tab, data[i], cursors[i % lanes]);
    state8 = _mm256_load_si256((const __m256i*)st);
  }

  // Main loop: unpack 160 symbols at a time, process 20 groups of 8.
  // digit_buf[d] = 32 bytes of digit d for all 32 input bytes.
  alignas(32) uint8_t digit_buf[5 * 32 + 4];  // +4 padding for gather overread

  // Precompute gather offsets for each of 20 groups within a 160-symbol batch.
  // Symbol k within batch is digit_buf[k%5][k/5], at byte offset (k%5)*32 + k/5.
  alignas(32) int32_t gather_offsets[20][8];
  for (int g = 0; g < 20; g++)
    for (int lane = 0; lane < lanes; lane++) {
      const int s = g * lanes + lane;
      gather_offsets[g][lane] = (s % 5) * 32 + s / 5;
    }
  const __m256i byte_mask = _mm256_set1_epi32(0xff);

  for (int64_t batch_base = full_batches - batch; batch_base >= 0; batch_base -= batch) {
    // Unpack 32 bytes into digit_buf via AVX2 divide-by-3
    int d = 0;
    unpack_ternary_avx2(
        _mm256_loadu_si256((const __m256i*)(data.bytes().data() + batch_base / 5)),
        [&](const __m256i rlo, const __m256i rhi) {
      _mm256_store_si256((__m256i*)(digit_buf + 32 * d++), _mm256_packus_epi16(rlo, rhi));
    });

    // Process 20 groups of 8 in reverse, gathering symbols via precomputed offsets
    for (int g = 20 - 1; g >= 0; g--) {
      const __m256i off8 = _mm256_load_si256((const __m256i*)gather_offsets[g]);
      const __m256i sym8 = _mm256_and_si256(
          _mm256_i32gather_epi32((const int*)digit_buf, off8, 1), byte_mask);

      // Shuffle per-symbol values
      const __m256i xmax8 = shuffle3(xmax_tab, sym8);
      const __m256i freq8 = shuffle3(freq_tab, sym8);
      const __m256i cum8 = shuffle3(cum_tab, sym8);
      const __m256i rcp8 = shuffle3(rcp_tab, sym8);
      const __m256i shift8 = shuffle3(shift_tab, sym8);

      // Renorm: emit low bytes and shift for lanes where state >= xmax
      for (;;) {
        const __m256i need = _mm256_andnot_si256(
            _mm256_cmpgt_epi32(xmax8, state8), _mm256_set1_epi32(-1));
        int mask = _mm256_movemask_epi8(need);
        if (!mask) break;
        // Emit one byte per needing lane, highest lane first (LIFO order)
        while (mask) {
          const int lane = (31 - __builtin_clz(mask)) / 4;
          const uint32_t val = _mm256_cvtsi256_si32(
              _mm256_permutevar8x32_epi32(state8, _mm256_set1_epi32(lane)));
          *cursors[lane]++ = val & 0xff;
          mask &= ~(0xf << (lane * 4));
        }
        state8 = _mm256_blendv_epi8(state8, _mm256_srli_epi32(state8, 8), need);
      }

      // SIMD encode: q = state / freq via add-and-shift reciprocal
      const __m256i hi8 = mulhi_epu32(state8, rcp8);
      const __m256i t8 = _mm256_srli_epi32(_mm256_sub_epi32(state8, hi8), 1);
      __m256i q8 = _mm256_srlv_epi32(_mm256_add_epi32(hi8, t8), shift8);
      // Fix freq=1 lanes (reciprocal is zero, correct q is state itself)
      q8 = _mm256_blendv_epi8(q8, state8, _mm256_cmpeq_epi32(freq8, one8));
      // r = state - q * freq; new_state = (q << total_bits) + r + cum
      const __m256i r8 = _mm256_sub_epi32(state8, _mm256_mullo_epi32(q8, freq8));
      state8 = _mm256_add_epi32(_mm256_add_epi32(_mm256_slli_epi32(q8, total_bits), r8), cum8);
    }  // groups of 8
  }  // batches of 160

  // Flush final state and reverse each buffer
  {
    alignas(32) uint32_t st[8];
    _mm256_store_si256((__m256i*)st, state8);
    for (int lane = 0; lane < lanes; lane++) {
      for (int j = 0; j < 4; j++) { *cursors[lane]++ = st[lane] & 0xff; st[lane] >>= 8; }
      const int len = int(cursors[lane] - buf_storage[lane].data());
      reverse(buf_storage[lane].data(), buf_storage[lane].data() + len);
    }
  }

  return finish();
}

// AVX2 decoder
static ternaries_t arithmetic_decode_avx2(const rans_freq_t& tab, const uint64_t n,
                                          const uint8_t* ptrs[lanes],
                                          const uint8_t* ends[lanes],
                                          uint32_t states[lanes]) {
  ternaries_t result(n);
  __m256i state8 = _mm256_loadu_si256((const __m256i*)states);
  const __m256i mask_total = _mm256_set1_epi32(rans_total - 1);
  const __m256i cum1 = _mm256_set1_epi32(tab.cum[1]);
  const __m256i cum2 = _mm256_set1_epi32(tab.cum[2]);
  const __m256i freq_tab = make_table8(tab.freq[0], tab.freq[1], tab.freq[2]);
  const __m256i cum_tab = make_table8(tab.cum[0], tab.cum[1], tab.cum[2]);

  ternary_writer_t writer(result);
  uint64_t i = 0;

  for (; i + lanes <= n; i += lanes) {
    // Slot and symbol lookup
    const __m256i slot8 = _mm256_and_si256(state8, mask_total);
    const __m256i sym8 = _mm256_add_epi32(_mm256_set1_epi32(2),
                          _mm256_add_epi32(_mm256_cmpgt_epi32(cum1, slot8),
                                           _mm256_cmpgt_epi32(cum2, slot8)));

    // Shuffle freq and cum per lane
    const __m256i freq8 = _mm256_permutevar8x32_epi32(freq_tab, sym8);
    const __m256i gcum8 = _mm256_permutevar8x32_epi32(cum_tab, sym8);

    // Decode
    state8 = _mm256_add_epi32(_mm256_sub_epi32(
                 _mm256_mullo_epi32(freq8, _mm256_srli_epi32(state8, total_bits)), gcum8), slot8);

    // Extract symbols and write
    alignas(32) int32_t syms[8];
    _mm256_store_si256((__m256i*)syms, sym8);
    for (int lane = 0; lane < lanes; lane++)
      writer.put(syms[lane]);

    // Renorm: read bytes and shift up for lanes where state < rans_lower
    {
      const __m256i lower8 = _mm256_set1_epi32(rans_lower);
      const __m256i lane_ids = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
      for (;;) {
        const __m256i need = _mm256_cmpgt_epi32(lower8, state8);
        int mask = _mm256_movemask_epi8(need);
        if (!mask) break;
        __m256i got8 = _mm256_setzero_si256();
        __m256i bytes8 = _mm256_setzero_si256();
        while (mask) {
          const int lane = __builtin_ctz(mask) / 4;
          if (ptrs[lane] < ends[lane]) {
            const __m256i lane_mask = _mm256_cmpeq_epi32(_mm256_set1_epi32(lane), lane_ids);
            bytes8 = _mm256_or_si256(bytes8,
                _mm256_and_si256(_mm256_set1_epi32(*ptrs[lane]++), lane_mask));
            got8 = _mm256_or_si256(got8, lane_mask);
          }
          mask &= ~(0xf << (lane * 4));
        }
        if (_mm256_testz_si256(got8, got8)) break;
        state8 = _mm256_blendv_epi8(state8,
            _mm256_or_si256(_mm256_slli_epi32(state8, 8), bytes8), got8);
      }
    }
  }

  // Tail
  alignas(32) uint32_t st[8];
  _mm256_store_si256((__m256i*)st, state8);
  for (; i < n; i++) {
    const int lane = int(i % lanes);
    writer.put(rans_get(st[lane], tab, ptrs[lane], ends[lane]));
  }
  writer.flush();
  return result;
}

#endif  // PENTAGO_SSE

}  // namespace

arithmetic_t arithmetic_encode(const ternaries_t data) {
  const auto counts = data.counts();
  const auto tab = make_freq(counts);
  const uint64_t n = data.size;

#if PENTAGO_SSE
  return arithmetic_encode_avx2(data, tab, counts);
#endif

  // Scalar fallback
  uint32_t states[lanes];
  for (int lane = 0; lane < lanes; lane++) states[lane] = rans_lower;
  const int buf_capacity = int(n / lanes * 2) + 64;
  Array<uint8_t> buf_storage[lanes];
  uint8_t* cursors[lanes];
  for (int lane = 0; lane < lanes; lane++) {
    buf_storage[lane] = Array<uint8_t>(buf_capacity, uninit);
    cursors[lane] = buf_storage[lane].data();
  }

  for (int64_t i = int64_t(n) - 1; i >= 0; i--)
    rans_put(states[i % lanes], tab, data[i], cursors[i % lanes]);

  // Flush and reverse
  for (int lane = 0; lane < lanes; lane++) {
    for (int j = 0; j < 4; j++) { *cursors[lane]++ = states[lane] & 0xff; states[lane] >>= 8; }
    const int len = int(cursors[lane] - buf_storage[lane].data());
    reverse(buf_storage[lane].data(), buf_storage[lane].data() + len);
  }

  const uint8_t* ptrs[lanes];
  int lengths[lanes];
  for (int lane = 0; lane < lanes; lane++) {
    ptrs[lane] = buf_storage[lane].data();
    lengths[lane] = int(cursors[lane] - buf_storage[lane].data());
  }
  return concat_streams(counts, ptrs, lengths);
}

ternaries_t arithmetic_decode(const arithmetic_t encoded) {
  GEODE_ASSERT(encoded.version == 2);
  const auto tab = make_freq(encoded.counts);
  const uint64_t n = encoded.total();

  // Validate stream lengths
  {
    uint64_t total_len = 0;
    for (int lane = 0; lane < lanes; lane++)
      total_len += encoded.stream_lengths[lane];
    GEODE_ASSERT(total_len == uint64_t(encoded.data.size()));
  }

  // Split streams and read initial state
  static const uint8_t dummy = 0;
  uint32_t states[lanes];
  const uint8_t* ptrs[lanes];
  const uint8_t* ends[lanes];
  int offset = 0;
  for (int lane = 0; lane < lanes; lane++) {
    const int len = encoded.stream_lengths[lane];
    const uint8_t* base = len ? encoded.data.data() + offset : &dummy;
    const uint8_t* end = base + len;
    offset += len;
    states[lane] = 0;
    const uint8_t* p = base;
    for (int j = 0; j < 4 && p < end; j++)
      states[lane] = (states[lane] << 8) | *p++;
    ptrs[lane] = p;
    ends[lane] = end;
  }

#if PENTAGO_SSE
  if (n >= lanes)
    return arithmetic_decode_avx2(tab, n, ptrs, ends, states);
#endif

  // Scalar fallback
  ternaries_t result(n);
  ternary_writer_t writer(result);
  for (uint64_t i = 0; i < n; i++) {
    const int lane = int(i % lanes);
    writer.put(rans_get(states[lane], tab, ptrs[lane], ends[lane]));
  }
  writer.flush();
  return result;
}

}  // namespace pentago
