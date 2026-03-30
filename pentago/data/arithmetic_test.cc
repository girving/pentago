// Arithmetic coding tests

#include "pentago/data/arithmetic.h"
#include "pentago/utility/log.h"
#include "pentago/utility/portable_hash.h"
#include "pentago/utility/random.h"
#include "pentago/utility/range.h"
#include "pentago/utility/test_assert.h"
#include "pentago/utility/threefry.h"
#include "gtest/gtest.h"
#include <chrono>
#include <cmath>
namespace pentago {
namespace {

using std::log2;

ternaries_t random_ternaries(Random& random, const int n, const Vector<uint64_t,3> weights) {
  const uint64_t total = weights[0] + weights[1] + weights[2];
  ternaries_t t(n);
  for (const int i : range(n)) {
    const uint64_t r = random.uniform<uint64_t>(total);
    t.set(i, r < weights[0] ? 0 : r < weights[0] + weights[1] ? 1 : 2);
  }
  return t;
}

void roundtrip(const ternaries_t input) {
  const auto encoded = arithmetic_encode(input);
  PENTAGO_ASSERT_EQ(encoded.total(), input.size);
  const auto decoded = arithmetic_decode(encoded);
  PENTAGO_ASSERT_EQ(decoded.size, input.size);
  for (const uint64_t i : range(input.size))
    PENTAGO_ASSERT_EQ(decoded[i], input[i]);
}

TEST(arithmetic, roundtrip_uniform) {
  Random random(42);
  // Encode 8 symbols (one per lane), then decode — simplest multi-symbol test
  roundtrip(random_ternaries(random, 10000, vec<uint64_t>(1, 1, 1)));
}

TEST(arithmetic, roundtrip_skewed) {
  Random random(123);
  roundtrip(random_ternaries(random, 10000, vec<uint64_t>(100, 50, 1)));
}

TEST(arithmetic, all_same) {
  for (const int v : {0, 1, 2}) {
    ternaries_t t(1000);
    for (const int i : range(1000))
      t.set(i, v);
    roundtrip(t);
  }
}

TEST(arithmetic, single_symbol) {
  for (const int v : {0, 1, 2}) {
    ternaries_t t(1);
    t.set(0, v);
    roundtrip(t);
  }
}

TEST(arithmetic, small_sizes) {
  Random random(77);
  for (const int n : range(20))
    roundtrip(random_ternaries(random, n, vec<uint64_t>(1, 1, 1)));
}

TEST(arithmetic, large_weights) {
  Random random(789);
  roundtrip(random_ternaries(random, 10000, vec<uint64_t>(700000, 200000, 100000)));
}

TEST(arithmetic, zero_weights) {
  for (const int v : {0, 1, 2}) {
    ternaries_t t(1000);
    for (const int i : range(1000))
      t.set(i, v);
    roundtrip(t);
  }
}

TEST(arithmetic, entropy) {
  const auto weights = vec<uint64_t>(70, 20, 10);
  const double total = 100.0;
  const int n = 100000;

  Random random(456);
  const auto input = random_ternaries(random, n, weights);
  const auto encoded = arithmetic_encode(input);

  double entropy = 0;
  for (const int i : range(3)) {
    const double p = weights[i] / total;
    entropy -= p * log2(p);
  }
  const double theoretical_bytes = entropy * n / 8;
  const double actual_bytes = encoded.data.size();
  const double ratio = actual_bytes / theoretical_bytes;
  slog("entropy test: theoretical %.0f bytes, actual %d bytes, ratio %.4f",
       theoretical_bytes, encoded.data.size(), ratio);
  ASSERT_GT(ratio, 0.99);
  ASSERT_LT(ratio, 1.02);  // slightly more slack for 8-stream overhead
}

TEST(arithmetic, determinism) {
  // Verify encoded output matches known hashes for portability across platforms
  const auto check = [](const ternaries_t t, const string& expected) {
    ASSERT_EQ(expected, sha1(arithmetic_encode(t).data)) << "size " << t.size;
  };
  // Repeating pattern
  {
    ternaries_t t(20);
    for (const int i : range(20))
      t.set(i, i % 3);
    check(t, "214a767752ee0ed88e361b42abc90ee05a3be0d1");
  }
  // All zeros
  check(ternaries_t(100), "523749d0496f43a7f8e33bad0c7db2e6edc5b8a6");
  // Skewed
  {
    ternaries_t t(1000);
    for (const int i : range(1000))
      t.set(i, int(threefry(7, i) % 3));
    check(t, "b3c12598e8201ef562e982f390ae7e6f4fad91af");
  }
}

TEST(arithmetic, finish_overflow) {
  // Test many patterns that could push lo into [0xff000000, 0xffffffff]
  // where the old (lo >> 24) + 1 would overflow
  Random random(999);
  for (const int _ : range(100)) {
    (void)_;
    const int n = random.uniform<int>(1, 200);
    ternaries_t t(n);
    for (const int i : range(n))
      t.set(i, random.uniform<int>(3));
    roundtrip(t);
  }
  // Specifically test heavily skewed data (high lo values)
  for (const int n : {1, 2, 7, 8, 9, 15, 16, 17, 100, 200}) {
    ternaries_t t(n);
    for (const int i : range(n))
      t.set(i, 2);  // all symbol 2 pushes lo high
    roundtrip(t);
  }
}

TEST(arithmetic, malformed_empty_data) {
  // Nonzero counts but no compressed data — streams are all empty
  const Vector<uint32_t,8> no_lens;
  const arithmetic_t bad{2, vec<uint64_t>(10, 10, 10), no_lens, Array<const uint8_t>()};
  const auto decoded = arithmetic_decode(bad);
  PENTAGO_ASSERT_EQ(decoded.size, 30);
  for (const uint64_t i : range(decoded.size))
    ASSERT_LT(decoded[i], 3);
}

TEST(arithmetic, malformed_zero_counts) {
  const Vector<uint32_t,8> no_lens;
  const arithmetic_t bad{2, vec<uint64_t>(0, 0, 0), no_lens, Array<const uint8_t>()};
  const auto decoded = arithmetic_decode(bad);
  PENTAGO_ASSERT_EQ(decoded.size, 0);
}

TEST(arithmetic, malformed_garbage) {
  // Valid stream_lengths summing to data size, but garbage content
  Array<uint8_t> garbage(256, uninit);
  for (const int i : range(garbage.size()))
    garbage[i] = uint8_t(threefry(222, i));
  Vector<uint32_t,8> lens;
  for (int i = 0; i < 8; i++) lens[i] = 32;
  const arithmetic_t bad{2, vec<uint64_t>(100, 100, 100), lens, garbage};
  const auto decoded = arithmetic_decode(bad);
  for (const uint64_t i : range(decoded.size))
    ASSERT_LT(decoded[i], 3);
}

TEST(arithmetic, malformed_truncated) {
  // Encode valid data, then truncate — stream_lengths won't match
  Random random(111);
  const auto input = random_ternaries(random, 1000, vec<uint64_t>(1, 1, 1));
  const auto encoded = arithmetic_encode(input);
  const int half = encoded.data.size() / 2;
  Array<uint8_t> truncated(half, uninit);
  for (const int i : range(half))
    truncated[i] = encoded.data[i];
  const arithmetic_t bad{2, encoded.counts, encoded.stream_lengths, truncated};
  ASSERT_THROW(arithmetic_decode(bad), RuntimeError);
}

TEST(arithmetic, malformed_overflow_counts) {
  const Vector<uint32_t,8> no_lens;
  const arithmetic_t bad{2, vec(UINT64_MAX, UINT64_MAX, uint64_t(1)), no_lens, Array<const uint8_t>()};
  ASSERT_THROW(arithmetic_decode(bad), RuntimeError);
}

TEST(arithmetic, malformed_stream_lengths) {
  Array<uint8_t> data(32);
  data.fill(0);
  // Lengths that don't sum to data size
  Vector<uint32_t,8> big_lens;
  for (int i = 0; i < 8; i++) big_lens[i] = 1000;
  const arithmetic_t bad1{2, vec<uint64_t>(10, 10, 10), big_lens, data};
  ASSERT_THROW(arithmetic_decode(bad1), RuntimeError);
  // Lengths that overflow uint64 when summed
  Vector<uint32_t,8> huge_lens;
  for (int i = 0; i < 8; i++) huge_lens[i] = UINT32_MAX / 4;
  const arithmetic_t bad2{2, vec<uint64_t>(10, 10, 10), huge_lens, data};
  ASSERT_THROW(arithmetic_decode(bad2), RuntimeError);
}

TEST(arithmetic, malformed_huge_counts) {
  // Individual count large enough that count * 0x3fff would overflow uint64
  const Vector<uint32_t,8> no_lens;
  const arithmetic_t bad{2, vec(uint64_t(1) << 52, uint64_t(1), uint64_t(1)), no_lens, Array<const uint8_t>()};
  ASSERT_THROW(arithmetic_decode(bad), RuntimeError);
}

TEST(arithmetic, one_per_lane) {
  // Exactly 8 symbols — one per lane
  ternaries_t t(8);
  for (const int i : range(8))
    t.set(i, i % 3);
  roundtrip(t);
}

TEST(arithmetic, fewer_than_lanes) {
  // 1..7 symbols — some lanes get no symbols
  for (const int n : range(1, 8)) {
    ternaries_t t(n);
    for (const int i : range(n))
      t.set(i, i % 3);
    roundtrip(t);
  }
}

TEST(arithmetic, extremely_skewed_counts) {
  // One symbol dominates — tests make_freq adjustment when two freqs are forced to 1
  for (const int dom : {0, 1, 2}) {
    ternaries_t t(100000);
    for (const int i : range(100000))
      t.set(i, dom);
    // Sprinkle a few of the other symbols
    t.set(0, (dom + 1) % 3);
    t.set(1, (dom + 2) % 3);
    roundtrip(t);
  }
}

TEST(arithmetic, two_zero_counts) {
  // Only one symbol present
  for (const int v : {0, 1, 2}) {
    for (const int n : {1, 8, 100}) {
      ternaries_t t(n);
      for (const int i : range(n))
        t.set(i, v);
      roundtrip(t);
    }
  }
}

TEST(arithmetic, large_n) {
  Random random(333);
  roundtrip(random_ternaries(random, 100000, vec<uint64_t>(1, 1, 1)));
}

TEST(arithmetic, benchmark) {
  // Benchmark encode/decode with skewed distribution (p ≈ 0.6, 0.3, 0.1)
  // Target ~1 second total.
  //
  // Results (10M symbols, p={60,30,10}, min of 10 iters):
  //                          encode      decode
  //   2026mar30 scalar:       36.4 M/s    77.1 M/s
  //   2026mar30 AVX2:        222.4 M/s   203.5 M/s
  //   2026mar30 opt:         296.3 M/s   305.5 M/s
  //   AVX2/scalar speedup:     8.1x        4.0x
  const int n = 10000000;
  const auto weights = vec<uint64_t>(60, 30, 10);
  const double total_w = 100.0;

  // Generate
  ternaries_t data(n);
  data.fill_random(42, Vector<uint16_t,2>(uint16_t(0.6 * 65536), uint16_t(0.9 * 65536)));

  // Run 10 iterations, take min time for each phase
  const int iters = 10;
  double best_enc = 1e9, best_dec = 1e9;
  const auto encoded = arithmetic_encode(data);
  for (int iter = 0; iter < iters; iter++) {
    auto t0 = std::chrono::high_resolution_clock::now();
    arithmetic_encode(data);
    auto t1 = std::chrono::high_resolution_clock::now();
    arithmetic_decode(encoded);
    auto t2 = std::chrono::high_resolution_clock::now();
    best_enc = std::min(best_enc, std::chrono::duration<double, std::milli>(t1 - t0).count());
    best_dec = std::min(best_dec, std::chrono::duration<double, std::milli>(t2 - t1).count());
  }

  // Verify roundtrip
  const auto decoded = arithmetic_decode(encoded);
  PENTAGO_ASSERT_EQ(decoded.size, data.size);
  for (int i = 0; i < 1000; i++) {
    const int j = int(uint64_t(threefry(99, i)) % n);
    PENTAGO_ASSERT_EQ(decoded[j], data[j]);
  }

  // Check compression vs entropy
  double entropy = 0;
  for (int i = 0; i < 3; i++) {
    const double p = weights[i] / total_w;
    entropy -= p * log2(p);
  }
  const double theoretical = entropy * n / 8;
  const double actual = encoded.data.size();
  const double ratio = actual / theoretical;

  slog("benchmark: n=%d, entropy=%.4f bits/sym, %d iters (min time)", n, entropy, iters);
  slog("  encode:   %.1f ms (%.1f M sym/s)", best_enc, n / 1e3 / best_enc);
  slog("  decode:   %.1f ms (%.1f M sym/s)", best_dec, n / 1e3 / best_dec);
  slog("  size: %.0f bytes (%.2fx entropy)", actual, ratio);
  ASSERT_GT(ratio, 0.99);
  ASSERT_LT(ratio, 1.02);
}

}  // namespace
}  // namespace pentago
