// Arithmetic coding of ternary arrays using 8 interleaved streams
//
// Format: 8 independent byte-renormalized arithmetic coders whose output bytes
// are interleaved: [s0_b0][s1_b0]...[s7_b0][s0_b1][s1_b1]..., padded to 8 bytes.

#include "pentago/data/arithmetic.h"
#include "pentago/utility/debug.h"
#include "pentago/utility/range.h"
#include <algorithm>
#include <cassert>
namespace pentago {

using std::max;

static const int lanes = 8;

static Vector<uint32_t,3> make_cumulative(const Vector<uint64_t,3> counts) {
  // Check for overflow, then renormalize so total fits in 14 bits with each weight at least 1
  constexpr auto M = numeric_limits<uint64_t>::max();
  GEODE_ASSERT(counts[0] <= M - counts[1] && counts[0] + counts[1] <= M - counts[2]);
  const uint64_t total = counts[0] + counts[1] + counts[2];
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

  // Encode with 8 interleaved streams
  stream_t streams[lanes];
  // Temporary per-stream output buffers
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

  // Interleave: find max buffer length, pad shorter ones
  int max_len = 0;
  for (int lane = 0; lane < lanes; lane++)
    max_len = max(max_len, int(buffers[lane].size()));
  for (int lane = 0; lane < lanes; lane++)
    buffers[lane].resize(max_len, 0);

  // Write interleaved output
  Array<uint8_t> output(max_len * lanes, uninit);
  for (int j = 0; j < max_len; j++)
    for (int lane = 0; lane < lanes; lane++)
      output[j * lanes + lane] = buffers[lane][j];

  return {counts, output};
}

namespace {

// Single-stream decoder state
struct decode_stream_t {
  uint32_t lo = 0;
  uint32_t hi = 0xffffffff;
  uint32_t value = 0;
  const uint8_t* ptr;
  const uint8_t* end;

  uint8_t next() { return ptr < end ? *ptr++ : 0; }

  void init(const uint8_t* data, const uint8_t* end) {
    ptr = data;
    this->end = end;
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
  const auto cum = make_cumulative(encoded.counts);
  const uint64_t n = encoded.total();
  ternaries_t result(n);

  // De-interleave into per-stream buffers
  const int total_bytes = encoded.data.size();
  const int bytes_per_stream = total_bytes / lanes;
  Array<uint8_t> buffers[lanes];
  for (int lane = 0; lane < lanes; lane++) {
    buffers[lane] = Array<uint8_t>(bytes_per_stream, uninit);
    for (int j = 0; j < bytes_per_stream; j++)
      buffers[lane][j] = encoded.data[j * lanes + lane];
  }

  // Initialize decoder streams
  decode_stream_t streams[lanes];
  for (int lane = 0; lane < lanes; lane++)
    streams[lane].init(buffers[lane].data(), buffers[lane].data() + bytes_per_stream);

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
