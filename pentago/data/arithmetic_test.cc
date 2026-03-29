// Arithmetic coding tests

#include "pentago/data/arithmetic.h"
#include "pentago/utility/log.h"
#include "pentago/utility/portable_hash.h"
#include "pentago/utility/random.h"
#include "pentago/utility/range.h"
#include "pentago/utility/test_assert.h"
#include "pentago/utility/threefry.h"
#include "gtest/gtest.h"
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
    check(t, "b4e206fd75925fc0e35024c02d55fb368288a928");
  }
  // All zeros
  check(ternaries_t(100), "94f86ed2756400378ab22ffd5b2937df942b7054");
  // Skewed
  {
    ternaries_t t(1000);
    for (const int i : range(1000))
      t.set(i, int(threefry(7, i) % 3));
    check(t, "a20a8481b336c4944055a5cb5e4cb454e41903b2");
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
  for (const int n : {1, 7, 8, 9, 15, 16, 17, 100, 200}) {
    ternaries_t t(n);
    for (const int i : range(n))
      t.set(i, 2);  // all symbol 2 pushes lo high
    roundtrip(t);
  }
}

TEST(arithmetic, malformed_empty_data) {
  // Decode with counts but no compressed data
  const arithmetic_t bad{vec<uint64_t>(10, 10, 10), Array<const uint8_t>()};
  const auto decoded = arithmetic_decode(bad);
  PENTAGO_ASSERT_EQ(decoded.size, 30);
  // Should not crash; values are garbage but bounded to {0,1,2}
  for (const uint64_t i : range(decoded.size))
    ASSERT_LT(decoded[i], 3);
}

TEST(arithmetic, malformed_truncated) {
  // Encode valid data, then truncate the compressed output
  Random random(111);
  const auto input = random_ternaries(random, 1000, vec<uint64_t>(1, 1, 1));
  const auto encoded = arithmetic_encode(input);
  // Truncate to half
  const int half = encoded.data.size() / 2;
  Array<uint8_t> truncated(half, uninit);
  for (const int i : range(half))
    truncated[i] = encoded.data[i];
  const arithmetic_t bad{encoded.counts, truncated};
  const auto decoded = arithmetic_decode(bad);
  // Should not crash; values bounded to {0,1,2}
  for (const uint64_t i : range(decoded.size))
    ASSERT_LT(decoded[i], 3);
}

TEST(arithmetic, malformed_garbage) {
  // Decode random garbage bytes
  Random random(222);
  Array<uint8_t> garbage(256, uninit);
  for (const int i : range(garbage.size()))
    garbage[i] = uint8_t(threefry(222, i));
  const arithmetic_t bad{vec<uint64_t>(100, 100, 100), garbage};
  const auto decoded = arithmetic_decode(bad);
  // Should not crash; values bounded to {0,1,2}
  for (const uint64_t i : range(decoded.size))
    ASSERT_LT(decoded[i], 3);
}

TEST(arithmetic, malformed_zero_counts) {
  // All counts zero
  const arithmetic_t bad{vec<uint64_t>(0, 0, 0), Array<const uint8_t>()};
  const auto decoded = arithmetic_decode(bad);
  PENTAGO_ASSERT_EQ(decoded.size, 0);
}

TEST(arithmetic, malformed_not_multiple_of_lanes) {
  // Compressed data size not a multiple of 8
  Array<uint8_t> odd(13, uninit);
  odd.fill(0);
  const arithmetic_t bad{vec<uint64_t>(5, 5, 5), odd};
  // bytes_per_stream = 13/8 = 1, which truncates — should not crash
  const auto decoded = arithmetic_decode(bad);
  for (const uint64_t i : range(decoded.size))
    ASSERT_LT(decoded[i], 3);
}

TEST(arithmetic, malformed_overflow_counts) {
  // Counts that overflow uint64_t when summed
  const arithmetic_t bad{vec(UINT64_MAX, UINT64_MAX, uint64_t(1)), Array<const uint8_t>()};
  ASSERT_THROW(arithmetic_decode(bad), RuntimeError);
}

}  // namespace
}  // namespace pentago
