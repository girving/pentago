// Arithmetic coding of ternary arrays using 8 concatenated streams
//
// Format: 8 independent byte-renormalized arithmetic coders whose output bytes
// are concatenated. Stream lengths are stored in arithmetic_t::stream_lengths.

#include "pentago/data/arithmetic.h"
#include "pentago/utility/debug.h"
#include "pentago/utility/range.h"
#include <algorithm>
#include <cassert>
#include <cstring>
namespace pentago {

using std::max;
using std::min;

static const int lanes = arithmetic_lanes;

static Vector<uint32_t,3> make_cumulative(const Vector<uint64_t,3> counts) {
  // Validate and renormalize so total fits in 14 bits with each weight at least 1
  constexpr uint64_t max_total = numeric_limits<uint64_t>::max() / 0x3fff;
  const uint64_t total = counts[0] + counts[1] + counts[2];
  GEODE_ASSERT(counts[0] <= total && counts[1] <= total && total <= max_total);
  const uint32_t w0 = total ? uint32_t(counts[0] * 0x3fff / total + 1) : 1;
  const uint32_t w1 = total ? uint32_t(counts[1] * 0x3fff / total + 1) : 1;
  const uint32_t w2 = total ? uint32_t(counts[2] * 0x3fff / total + 1) : 1;
  return vec(w0, w0 + w1, w0 + w1 + w2);
}

namespace {

// Single-stream encoder state
struct stream_t {
  uint32_t lo = 0;
  uint32_t hi = 0xffffffff;

  void encode(const Vector<uint32_t,3> cum, const int symbol) {
    assert(unsigned(symbol) < 3);
    const uint32_t total = cum[2];
    const uint64_t range = uint64_t(hi) - lo + 1;
    const uint32_t cum_lo = symbol > 0 ? cum[symbol - 1] : 0;
    const uint32_t cum_hi = cum[symbol];
    hi = lo + uint32_t(range * cum_hi / total) - 1;
    lo = lo + uint32_t(range * cum_lo / total);
  }

  // Returns number of bytes emitted (0, 1, or more via loop)
  int renormalize(uint8_t* out) {
    int n = 0;
    while ((lo ^ hi) < 0x01000000) {
      out[n++] = lo >> 24;
      lo <<= 8;
      hi = (hi << 8) | 0xff;
    }
    return n;
  }

  void finish(uint8_t* out, int& n) {
    // Emit a value in (lo, hi]. We use lo + half the interval, which avoids overflow.
    const uint32_t mid = lo + uint32_t((uint64_t(hi) - lo + 1) / 2);
    out[n++] = mid >> 24;
    out[n++] = mid >> 16;
    out[n++] = mid >> 8;
    out[n++] = mid;
  }
};

}  // namespace

arithmetic_t arithmetic_encode(const ternaries_t data) {
  // Count symbols
  Vector<uint64_t,3> counts(0, 0, 0);
  for (const uint64_t i : range(data.size))
    counts[data[i]]++;
  const auto cum = make_cumulative(counts);

  // Encode with 8 streams
  stream_t streams[lanes];
  vector<uint8_t> buffers[lanes];

  for (uint64_t i = 0; i < data.size; i++) {
    const int lane = i % lanes;
    streams[lane].encode(cum, data[i]);
    uint8_t bytes[8];
    const int n = streams[lane].renormalize(bytes);
    for (int j = 0; j < n; j++)
      buffers[lane].push_back(bytes[j]);
  }

  // Flush each stream
  for (int lane = 0; lane < lanes; lane++) {
    uint8_t bytes[4];
    int n = 0;
    streams[lane].finish(bytes, n);
    for (int j = 0; j < n; j++)
      buffers[lane].push_back(bytes[j]);
  }

  // Concatenate streams
  Vector<uint32_t,lanes> stream_lengths;
  uint64_t total_bytes = 0;
  for (int lane = 0; lane < lanes; lane++) {
    stream_lengths[lane] = CHECK_CAST_INT(uint64_t(buffers[lane].size()));
    total_bytes += stream_lengths[lane];
  }
  Array<uint8_t> output(CHECK_CAST_INT(total_bytes), uninit);
  int offset = 0;
  for (int lane = 0; lane < lanes; lane++) {
    memcpy(output.data() + offset, buffers[lane].data(), stream_lengths[lane]);
    offset += stream_lengths[lane];
  }

  return {1, counts, stream_lengths, output};
}

namespace {

// Single-stream decoder state
struct decode_stream_t {
  uint32_t lo = 0;
  uint32_t hi = 0xffffffff;
  uint32_t value = 0;
  RawArray<const uint8_t> data;
  int pos = 0;

  uint8_t next() { return pos < data.size() ? data[pos++] : 0; }

  void init(const RawArray<const uint8_t> data) {
    this->data = data;
    pos = 0;
    for (int i = 0; i < 4; i++)
      value = (value << 8) | next();
  }

  int decode(const Vector<uint32_t,3> cum) {
    const uint32_t total = cum[2];
    const uint64_t range = uint64_t(hi) - lo + 1;
    const uint32_t scaled = uint32_t((uint64_t(value) - lo) * total / range);
    const int symbol = scaled < cum[0] ? 0 : scaled < cum[1] ? 1 : 2;
    const uint32_t cum_lo = symbol > 0 ? cum[symbol - 1] : 0;
    const uint32_t cum_hi = cum[symbol];
    hi = lo + uint32_t(range * cum_hi / total) - 1;
    lo = lo + uint32_t(range * cum_lo / total);
    return symbol;
  }

  void renormalize() {
    while ((lo ^ hi) < 0x01000000) {
      lo <<= 8;
      hi = (hi << 8) | 0xff;
      value = (value << 8) | next();
    }
  }
};

}  // namespace

ternaries_t arithmetic_decode(const arithmetic_t encoded) {
  GEODE_ASSERT(encoded.version == 1);
  const auto cum = make_cumulative(encoded.counts);
  const uint64_t n = encoded.total();
  ternaries_t result(n);

  // Validate and initialize decoder streams from concatenated data
  decode_stream_t streams[lanes];
  {
    uint64_t total_len = 0;
    for (int lane = 0; lane < lanes; lane++)
      total_len += encoded.stream_lengths[lane];
    GEODE_ASSERT(total_len == uint64_t(encoded.data.size()));
  }
  int offset = 0;
  for (int lane = 0; lane < lanes; lane++) {
    const int len = encoded.stream_lengths[lane];
    streams[lane].init(encoded.data.slice(offset, offset + len));
    offset += len;
  }

  // Decode
  for (uint64_t i = 0; i < n; i++) {
    const int lane = i % lanes;
    const int symbol = streams[lane].decode(cum);
    streams[lane].renormalize();
    result.set(i, symbol);
  }

  return result;
}

}  // namespace pentago
