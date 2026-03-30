// rANS (range Asymmetric Numeral Systems) coding of ternary arrays
//
// Format (version 2): 8 independent rANS coders with 32-bit state, concatenated.
// Uses power-of-2 total for division-free decode. 8-bit renormalization.
// Each stream's bytes are stored reversed (encoder writes LIFO, decoder reads FIFO).

#include "pentago/data/arithmetic.h"
#include "pentago/utility/debug.h"
#include "pentago/utility/range.h"
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
  uint32_t cum[3];  // {0, freq[0], freq[0]+freq[1]}
  uint32_t x_max[3];  // renormalize threshold per symbol
};

}  // namespace

static rans_freq_t make_freq(const Vector<uint64_t,3> counts) {
  constexpr uint64_t max_total = numeric_limits<uint64_t>::max() / 0x3fff;
  const uint64_t sum = counts[0] + counts[1] + counts[2];
  GEODE_ASSERT(counts[0] <= sum && counts[1] <= sum && sum <= max_total);

  uint32_t f[3];
  if (sum) {
    // Assign proportionally, then fix up to ensure sum = rans_total and each >= 1
    for (int i = 0; i < 3; i++)
      f[i] = uint32_t(counts[i] * rans_total / sum);
    // Ensure each is at least 1, stealing from the largest
    for (int i = 0; i < 3; i++)
      if (!f[i]) f[i] = 1;
    // Adjust the largest to hit the target sum
    const int32_t adjust = int32_t(rans_total) - int32_t(f[0] + f[1] + f[2]);
    int largest = 0;
    if (f[1] > f[largest]) largest = 1;
    if (f[2] > f[largest]) largest = 2;
    GEODE_ASSERT(int32_t(f[largest]) + adjust >= 1);
    f[largest] += adjust;
  } else {
    f[0] = f[1] = f[2] = rans_total / 3;
    f[0] += rans_total - f[0] - f[1] - f[2];
  }
  const uint32_t xm = (rans_lower >> total_bits) << 8;
  return {{f[0], f[1], f[2]},
          {0, f[0], f[0] + f[1]},
          {xm * f[0], xm * f[1], xm * f[2]}};
}

namespace {

static void rans_put(uint32_t& state, const rans_freq_t& tab, const int sym,
                     vector<uint8_t>& out) {
  assert(unsigned(sym) < 3);
  while (state >= tab.x_max[sym]) {
    out.push_back(state & 0xff);
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

}  // namespace

arithmetic_t arithmetic_encode(const ternaries_t data) {
  Vector<uint64_t,3> counts(0, 0, 0);
  {
    ternary_reader_t reader(data);
    for (uint64_t i = 0; i < data.size; i++)
      counts[reader.next()]++;
  }
  const auto tab = make_freq(counts);

  // Read all symbols for reverse access
  const uint64_t n = data.size;
  Array<uint8_t> symbols(CHECK_CAST_INT(n), uninit);
  {
    ternary_reader_t reader(data);
    for (uint64_t i = 0; i < n; i++)
      symbols[int(i)] = reader.next();
  }

  // Encode in reverse
  uint32_t states[lanes];
  for (int lane = 0; lane < lanes; lane++) states[lane] = rans_lower;
  vector<uint8_t> buffers[lanes];

  for (int64_t i = int64_t(n) - 1; i >= 0; i--)
    rans_put(states[i % lanes], tab, symbols[int(i)], buffers[i % lanes]);

  // Flush final state (little-endian, 4 bytes)
  for (int lane = 0; lane < lanes; lane++) {
    auto& buf = buffers[lane];
    for (int j = 0; j < 4; j++) {
      buf.push_back(states[lane] & 0xff);
      states[lane] >>= 8;
    }
    // Reverse so decoder can read forward
    reverse(buf.begin(), buf.end());
  }

  // Concatenate
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

  return {2, counts, stream_lengths, output};
}

ternaries_t arithmetic_decode(const arithmetic_t encoded) {
  GEODE_ASSERT(encoded.version == 2);
  const auto tab = make_freq(encoded.counts);
  const uint64_t n = encoded.total();
  ternaries_t result(n);

  // Validate stream lengths
  {
    uint64_t total_len = 0;
    for (int lane = 0; lane < lanes; lane++)
      total_len += encoded.stream_lengths[lane];
    GEODE_ASSERT(total_len == uint64_t(encoded.data.size()));
  }

  // Split streams, read initial state from first 4 bytes (big-endian after reverse)
  // Use a dummy byte for empty streams to avoid null pointer arithmetic
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

  // Decode forward
  ternary_writer_t writer(result);
  for (uint64_t i = 0; i < n; i++) {
    const int lane = int(i % lanes);
    writer.put(rans_get(states[lane], tab, ptrs[lane], ends[lane]));
  }
  writer.flush();

  return result;
}

}  // namespace pentago
