// Tests for shard_permute modular Feistel permutation

#include "pentago/data/shard_permute.h"
#include "pentago/data/shard.h"
#include "pentago/utility/log.h"
#include "pentago/utility/random.h"
#include "pentago/utility/range.h"
#include "pentago/utility/test_assert.h"
#include "gtest/gtest.h"
#include <unordered_set>
namespace pentago {
namespace {

using std::unordered_set;

// Verify hardcoded n values match shard_mapping_t(slice).total()
TEST(shard_permute, constants_match) {
  for (const int s : range(19)) {
    const shard_mapping_t m(s);
    const auto& c = permute_constants[s];
    PENTAGO_ASSERT_EQ(c.n, m.total());
    const uint32_t b = uint32_t(c.n / c.a);
    // Verify a * b <= n
    const uint64_t ab = uint64_t(c.a) * b;
    PENTAGO_ASSERT_LE(ab, c.n);
    // Verify a and b are both >= 2
    PENTAGO_ASSERT_GE(c.a, 2u);
    PENTAGO_ASSERT_GE(b, 2u);
    const uint64_t dropped = c.n - ab;
    slog("slice %d: n=%llu, a=%u, b=%u, dropped=%llu", s, c.n, c.a, b, dropped);
  }
}

// Exhaustive roundtrip for small slices: forward(inverse(x)) == x and inverse(forward(x)) == x
TEST(shard_permute, exhaustive_roundtrip) {
  for (const int s : range(5)) {
    const shard_permute_t p(s);
    const uint64_t n = permute_constants[s].n;
    for (const uint64_t x : range(n)) {
      const uint64_t y = p.forward(x);
      PENTAGO_ASSERT_LT(y, n);
      PENTAGO_ASSERT_EQ(p.inverse(y), x);
      PENTAGO_ASSERT_EQ(p.forward(p.inverse(x)), x);
    }
    slog("slice %d: exhaustive roundtrip passed (n=%llu)", s, n);
  }
}

// Exhaustive bijection for small slices: all forward(x) values are distinct
TEST(shard_permute, exhaustive_bijection) {
  for (const int s : range(5)) {
    const shard_permute_t p(s);
    const uint64_t n = permute_constants[s].n;
    unordered_set<uint64_t> seen;
    for (const uint64_t x : range(n)) {
      const uint64_t y = p.forward(x);
      ASSERT_TRUE(seen.insert(y).second)
          << "collision at slice " << s << " x=" << x << " y=" << y;
    }
    PENTAGO_ASSERT_EQ(seen.size(), n);
    slog("slice %d: exhaustive bijection passed (n=%llu)", s, n);
  }
}

// Verify identity mapping for dropped entries (x in [a*b, n))
TEST(shard_permute, identity_for_dropped) {
  for (const int s : range(19)) {
    const shard_permute_t p(s);
    const auto& c = permute_constants[s];
    const uint64_t ab = uint64_t(c.a) * uint32_t(c.n / c.a);
    int count = 0;
    for (uint64_t x = ab; x < c.n; x++) {
      PENTAGO_ASSERT_EQ(p.forward(x), x);
      PENTAGO_ASSERT_EQ(p.inverse(x), x);
      count++;
    }
    slog("slice %d: %d dropped entries identity-mapped", s, count);
  }
}

// Sample roundtrip for larger slices
TEST(shard_permute, sample_roundtrip) {
  Random random(uint128_t(0x243f6a8885a308d3));
  for (const int s : range(5, 19)) {
    const shard_permute_t p(s);
    const auto& c = permute_constants[s];
    for (const int i __attribute__((unused)) : range(10000)) {
      const uint64_t x = random.bits<uint64_t>() % c.n;
      const uint64_t y = p.forward(x);
      PENTAGO_ASSERT_LT(y, c.n);
      PENTAGO_ASSERT_EQ(p.inverse(y), x);
      PENTAGO_ASSERT_EQ(p.forward(p.inverse(x)), x);
    }
    slog("slice %d: sample roundtrip passed (n=%llu)", s, c.n);
  }
}

// Chi-squared bucket uniformity for consecutive blocks of 256 inputs.
// This tests the core scatter use case: one position's 256 rotations should spread
// uniformly across the output space, not cluster.
TEST(shard_permute, chi_squared_buckets) {
  Random random(uint128_t(0x3c6ef372fe94f82b));
  const int bins = 16;
  const int block = 256;
  const double expected = double(block) / bins;  // 16 per bin
  for (const int s : range(5, 19)) {
    const shard_permute_t p(s);
    const uint64_t ab = p.ab;
    const int trials = 1000;
    double chi2_sum = 0;
    for (const int t __attribute__((unused)) : range(trials)) {
      // Pick a random base in [0, ab - 256)
      const uint64_t base = random.bits<uint64_t>() % (ab - block);
      int counts[bins] = {};
      for (const int r : range(block)) {
        const uint64_t y = p.forward(base + r);
        counts[int(double(y) / ab * bins)]++;
      }
      double chi2 = 0;
      for (const int i : range(bins))
        chi2 += (counts[i] - expected) * (counts[i] - expected) / expected;
      chi2_sum += chi2;
    }
    const double mean_chi2 = chi2_sum / trials;
    // Expected chi-squared with bins-1=15 dof: mean=15, stddev=sqrt(30)≈5.5
    // With 1000 trials, mean of means has stddev ≈ 5.5/sqrt(1000) ≈ 0.17
    // Accept within ~6 sigma: [14, 16]
    slog("slice %d: mean chi2 = %.2f (expected 15)", s, mean_chi2);
    ASSERT_GT(mean_chi2, 14) << "slice " << s << ": outputs too uniform (degenerate permutation?)";
    ASSERT_LT(mean_chi2, 16) << "slice " << s << ": outputs too clustered";
  }
}

// Mean displacement: for a random permutation on [0, m), E[|forward(x) - x|] = m/3.
// Exhaustive for small slices, sampled for large ones.
TEST(shard_permute, mean_displacement) {
  Random random(uint128_t(0xd1310ba698dfb5ac));
  for (const int s : range(19)) {
    const shard_permute_t p(s);
    const uint64_t ab = p.ab;
    double displacement_sum = 0;
    uint64_t count = 0;
    if (s < 5) {
      // Exhaustive over [0, ab)
      for (const uint64_t x : range(ab)) {
        const uint64_t y = p.forward(x);
        displacement_sum += double(y > x ? y - x : x - y);
        count++;
      }
    } else {
      // Sample 100000 values from [0, ab)
      count = 100000;
      for (const uint64_t i __attribute__((unused)) : range(count)) {
        const uint64_t x = random.bits<uint64_t>() % ab;
        const uint64_t y = p.forward(x);
        displacement_sum += double(y > x ? y - x : x - y);
      }
    }
    const double mean_disp = displacement_sum / count;
    const double expected = double(ab) / 3.0;
    const double ratio = mean_disp / expected;
    slog("slice %d: mean displacement = %.0f, expected = %.0f, ratio = %.4f",
         s, mean_disp, expected, ratio);
    // Exhaustive slices have no sampling error; sampled slices have small error at 100K samples.
    // All observed ratios are within [0.995, 1.02], so [0.97, 1.03] is generous.
    ASSERT_GT(ratio, 0.97) << "slice " << s << ": displacement too small";
    ASSERT_LT(ratio, 1.03) << "slice " << s << ": displacement too large";
  }
}

#if PENTAGO_SSE
// Helper: check that forward8/inverse8 match scalar for 8 specific values
static void check_forward8(const shard_permute_t& p, const uint64_t xv[8]) {
  alignas(32) uint64_t in[8], out[8];
  for (int j = 0; j < 8; j++) in[j] = xv[j];
  const uint64x8 x = {_mm256_load_si256((const __m256i*)&in[0]),
                       _mm256_load_si256((const __m256i*)&in[4])};
  const auto y = p.forward8(x);
  _mm256_store_si256((__m256i*)&out[0], y.v0);
  _mm256_store_si256((__m256i*)&out[4], y.v1);
  for (int j = 0; j < 8; j++)
    PENTAGO_ASSERT_EQ(out[j], p.forward(in[j]));
}

static void check_inverse8(const shard_permute_t& p, const uint64_t yv[8]) {
  alignas(32) uint64_t in[8], out[8];
  for (int j = 0; j < 8; j++) in[j] = yv[j];
  const uint64x8 y = {_mm256_load_si256((const __m256i*)&in[0]),
                       _mm256_load_si256((const __m256i*)&in[4])};
  const auto x = p.inverse8(y);
  _mm256_store_si256((__m256i*)&out[0], x.v0);
  _mm256_store_si256((__m256i*)&out[4], x.v1);
  for (int j = 0; j < 8; j++)
    PENTAGO_ASSERT_EQ(out[j], p.inverse(in[j]));
}

// Verify AVX2 forward8/inverse8 match scalar, including boundary cases
TEST(shard_permute, avx2_forward_matches_scalar) {
  Random random(uint128_t(0x6a09e667bb67ae85));
  for (const int s : range(19)) {
    const shard_permute_t p(s);
    const auto& c = permute_constants[s];
    const uint64_t ab = p.ab;
    const uint32_t b = p.b;

    // Boundary cases: zero, near ab, dropped entries, divmod correction points
    uint64_t boundary[] = {
      0, 1, uint64_t(b) - 1, uint64_t(b), uint64_t(b) + 1,  // low end, divmod boundary
      ab - 1, ab > 0 ? ab - uint64_t(b) : 0,                 // near ab
      ab, c.n - 1,                                             // dropped entries
    };
    // Values near multiples of b trigger inv_b_d correction (q off by 1)
    for (uint64_t k = 1; k <= 8 && k * b < ab; k++) {
      const uint64_t vals[] = {k * b - 1, k * b, k * b + 1};
      for (const auto v : vals)
        if (v < c.n) {
          uint64_t xv[8];
          for (int j = 0; j < 8; j++) xv[j] = v;
          check_forward8(p, xv);
          check_inverse8(p, xv);
        }
    }
    // Test boundary values (fill remaining slots with 0)
    {
      uint64_t xv[8] = {};
      const int nb = std::min(int(sizeof(boundary) / sizeof(boundary[0])), 8);
      for (int j = 0; j < nb; j++) xv[j] = boundary[j];
      check_forward8(p, xv);
      check_inverse8(p, xv);
    }
    // Mix of boundary and random in the same batch
    {
      uint64_t xv[8];
      xv[0] = 0;
      xv[1] = ab - 1;
      xv[2] = ab < c.n ? ab : 0;  // first dropped (or 0 if none)
      xv[3] = c.n - 1;
      for (int j = 4; j < 8; j++) xv[j] = random.bits<uint64_t>() % c.n;
      check_forward8(p, xv);
      check_inverse8(p, xv);
    }

    // Random values
    for (const int i __attribute__((unused)) : range(1250)) {
      uint64_t xv[8];
      for (int j = 0; j < 8; j++)
        xv[j] = random.bits<uint64_t>() % c.n;
      check_forward8(p, xv);
    }
    slog("slice %d: avx2 forward matches scalar", s);
  }
}

TEST(shard_permute, avx2_inverse_matches_scalar) {
  Random random(uint128_t(0xbb67ae856a09e667));
  for (const int s : range(19)) {
    const shard_permute_t p(s);
    const auto& c = permute_constants[s];

    // Random values
    for (const int i __attribute__((unused)) : range(1250)) {
      uint64_t yv[8];
      for (int j = 0; j < 8; j++)
        yv[j] = random.bits<uint64_t>() % c.n;
      check_inverse8(p, yv);
    }
    slog("slice %d: avx2 inverse matches scalar", s);
  }
}
#endif  // PENTAGO_SSE

}  // namespace
}  // namespace pentago
