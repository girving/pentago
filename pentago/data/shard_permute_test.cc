// Tests for shard_permute L/H bit-level permutation

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

// Verify hardcoded n and k values match shard_mapping_t(slice).total()
TEST(shard_permute, constants_match) {
  for (const int s : range(19)) {
    const shard_mapping_t m(s);
    const auto& c = permute_constants[s];
    PENTAGO_ASSERT_EQ(c.n, m.total());
    const int k = 63 - __builtin_clzll(c.n);
    const uint64_t pow2k = uint64_t(1) << k;
    PENTAGO_ASSERT_LE(pow2k, c.n);
    PENTAGO_ASSERT_GT(pow2k * 2, c.n);
    for (const int i : range(PERM_STEPS)) {
      PENTAGO_ASSERT_GE(c.shifts[i], 1);
      PENTAGO_ASSERT_LT(c.shifts[i], k);
    }
    slog("slice %d: n=%llu, k=%d, overlap=%.1f%%", s, c.n, k,
         double(2 * pow2k - c.n) / c.n * 100);
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
// across the output space.  For slices with high L/H overlap the scatter is
// random-quality (chi2≈15).  Low-overlap slices have higher chi2 but still adequate
// for ML training with large batch sizes.
TEST(shard_permute, chi_squared_buckets) {
  Random random(uint128_t(0x3c6ef372fe94f82b));
  const int bins = 16;
  const int block = 256;
  const double expected = double(block) / bins;  // 16 per bin
  for (const int s : range(5, 19)) {
    const shard_permute_t p(s);
    const uint64_t n = p.n;
    const int trials = 1000;
    double chi2_sum = 0;
    for (const int t __attribute__((unused)) : range(trials)) {
      const uint64_t base = random.bits<uint64_t>() % (n - block);
      int counts[bins] = {};
      for (const int r : range(block)) {
        const uint64_t y = p.forward(base + r);
        counts[int(double(y) / n * bins)]++;
      }
      double chi2 = 0;
      for (const int i : range(bins))
        chi2 += (counts[i] - expected) * (counts[i] - expected) / expected;
      chi2_sum += chi2;
    }
    const double mean_chi2 = chi2_sum / trials;
    // Bound scales with L/H overlap: high overlap → chi2≈15, low → up to ~250.
    // Formula: 15/overlap capped at 250, with a floor of 18 for sampling noise.
    const uint64_t pow2k = uint64_t(1) << (63 - __builtin_clzll(n));
    const double overlap = double(2 * pow2k - n) / n;
    const double max_chi2 = std::min(250.0, std::max(18.0, 15.0 / overlap));
    slog("slice %d: mean chi2 = %.2f (max %.0f, overlap %.1f%%)",
         s, mean_chi2, max_chi2, overlap * 100);
    ASSERT_GT(mean_chi2, 10) << "slice " << s << ": outputs too uniform (degenerate permutation?)";
    ASSERT_LT(mean_chi2, max_chi2) << "slice " << s << ": outputs too clustered";
  }
}

// Batch diversity: simulate ML training by reading batch_size consecutive entries
// from a single shard and counting how many distinct positions they come from.
// With 131072 shards each position contributes 256/131072 < 0.002 entries per shard,
// so a batch of 256 should contain ~256 distinct positions (no collisions).
// Even with a poor permutation, collisions require two rotations of the same position
// to land in the same shard AND be adjacent within that shard, which is very unlikely.
TEST(shard_permute, batch_diversity) {
  Random random(uint128_t(0x3c6ef372fe94f82b));
  const int batch_size = 256;
  const int total_shards = 131072;  // realistic shard count
  const int shard_shift = __builtin_ctz(total_shards);
  const int shard_mask = total_shards - 1;
  const int trials = 200;
  for (const int s : range(19)) {
    const shard_permute_t p(s);
    const uint64_t n = p.n;
    // Skip slices too small to fill a batch within one shard
    const uint64_t min_shard_size = n >> shard_shift;
    if (min_shard_size < uint64_t(batch_size)) continue;
    double total_distinct = 0;
    int valid_trials = 0;
    for (const int t __attribute__((unused)) : range(trials)) {
      const int shard = random.bits<uint32_t>() & shard_mask;
      const uint64_t shard_size = (n + shard_mask - shard) >> shard_shift;
      if (shard_size < uint64_t(batch_size)) continue;
      const uint64_t start = random.bits<uint64_t>() % (shard_size - batch_size);

      // Read batch_size consecutive entries from this shard.
      // Each entry's shuffled index is (start + i) << shard_shift | shard.
      // Inverse-permute to get the original linear index, then divide by 256
      // to get the position (stripping the rotation).
      unordered_set<uint64_t> positions;
      for (const int i : range(batch_size)) {
        const uint64_t shuffled = ((start + i) << shard_shift) | shard;
        const uint64_t linear = p.inverse(shuffled);
        positions.insert(linear / 256);  // position = linear index / 256 rotations
      }
      total_distinct += positions.size();
      valid_trials++;
    }
    ASSERT_GT(valid_trials, 0) << "slice " << s << ": no valid trials";
    const double mean_distinct = total_distinct / valid_trials;
    // With 131072 shards and a perfect permutation, collisions are very rare
    // (256/256 distinct).  Low-overlap slices cluster some rotations into nearby
    // shuffled indices, reducing diversity slightly.  Thresholds are set per-slice
    // just below observed values.
    //   slices 8-10 (7-29% overlap): ~244-248 distinct
    //   all others:                   256 distinct
    const double min_distinct = s >= 8 && s <= 10 ? 243.0 : 255.0;
    slog("slice %d: mean distinct positions per batch = %.1f / %d (%d trials)",
         s, mean_distinct, batch_size, valid_trials);
    ASSERT_GT(mean_distinct, min_distinct)
        << "slice " << s << ": too many position collisions in training batches";
  }
}

// Mean displacement: for a random permutation on [0, m), E[|forward(x) - x|] = m/3.
// Exhaustive for small slices, sampled for large ones. Low-overlap slices may have
// lower displacement since some values are identity-mapped by both windows.
TEST(shard_permute, mean_displacement) {
  Random random(uint128_t(0xd1310ba698dfb5ac));
  for (const int s : range(19)) {
    const shard_permute_t p(s);
    const uint64_t n = p.n;
    double displacement_sum = 0;
    uint64_t count = 0;
    if (s < 5) {
      for (const uint64_t x : range(n)) {
        const uint64_t y = p.forward(x);
        displacement_sum += double(y > x ? y - x : x - y);
        count++;
      }
    } else {
      count = 100000;
      for (const uint64_t i __attribute__((unused)) : range(count)) {
        const uint64_t x = random.bits<uint64_t>() % n;
        const uint64_t y = p.forward(x);
        displacement_sum += double(y > x ? y - x : x - y);
      }
    }
    const double mean_disp = displacement_sum / count;
    const double expected = double(n) / 3.0;
    const double ratio = mean_disp / expected;
    slog("slice %d: mean displacement = %.0f, expected = %.0f, ratio = %.4f",
         s, mean_disp, expected, ratio);
    // Low-overlap slices have lower displacement (values stuck in one window).
    // Accept ratio >= 0.5 for all slices.
    ASSERT_GT(ratio, 0.50) << "slice " << s << ": displacement too small";
    ASSERT_LT(ratio, 1.10) << "slice " << s << ": displacement too large";
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

// Verify AVX2 forward8/inverse8 match scalar
TEST(shard_permute, avx2_forward_matches_scalar) {
  Random random(uint128_t(0x6a09e667bb67ae85));
  for (const int s : range(19)) {
    const shard_permute_t p(s);
    const auto& c = permute_constants[s];

    // Boundary cases: L/H window edges
    const uint64_t pow2k = p.pow2k;
    const uint64_t off = p.offset;
    uint64_t boundary[] = {
      0, 1, pow2k - 1, pow2k,
      off > 0 ? off - 1 : 0, off, off + 1,
      c.n - 1,
    };
    {
      uint64_t xv[8] = {};
      const int nb = std::min(int(sizeof(boundary) / sizeof(boundary[0])), 8);
      for (int j = 0; j < nb; j++) xv[j] = std::min(boundary[j], c.n - 1);
      check_forward8(p, xv);
      check_inverse8(p, xv);
    }
    // Mix of boundary and random
    {
      uint64_t xv[8];
      xv[0] = 0;
      xv[1] = pow2k - 1;
      xv[2] = off;
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

    // Boundary cases: L/H window edges (same as forward test)
    const uint64_t pow2k = p.pow2k;
    const uint64_t off = p.offset;
    uint64_t boundary[] = {
      0, 1, pow2k - 1, pow2k,
      off > 0 ? off - 1 : 0, off, off + 1,
      c.n - 1,
    };
    {
      uint64_t yv[8] = {};
      const int nb = std::min(int(sizeof(boundary) / sizeof(boundary[0])), 8);
      for (int j = 0; j < nb; j++) yv[j] = std::min(boundary[j], c.n - 1);
      check_inverse8(p, yv);
    }

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

// Fixed-output determinism: known inputs → known outputs, catches platform differences.
// Generated from slices 0, 9, 18 with specific inputs.
TEST(shard_permute, determinism) {
  // Slice 0 (n=256, k=8): forward and inverse of selected values
  {
    const shard_permute_t p(0);
    PENTAGO_ASSERT_EQ(p.forward(0), 0u);
    PENTAGO_ASSERT_EQ(p.forward(1), 92u);
    PENTAGO_ASSERT_EQ(p.forward(127), 127u);
    PENTAGO_ASSERT_EQ(p.forward(255), 222u);
    PENTAGO_ASSERT_EQ(p.inverse(0), 0u);
    PENTAGO_ASSERT_EQ(p.inverse(92), 1u);
    PENTAGO_ASSERT_EQ(p.inverse(222), 255u);
  }
  // Slice 9 (n=3550828544, k=31): mid-range slice
  {
    const shard_permute_t p(9);
    PENTAGO_ASSERT_EQ(p.forward(0), 0u);
    PENTAGO_ASSERT_EQ(p.forward(1000000), 485452711u);
    PENTAGO_ASSERT_EQ(p.forward(3550828543), 3450164501u);
    PENTAGO_ASSERT_EQ(p.inverse(485452711), 1000000u);
  }
  // Slice 18 (n=65007135675648, k=45): largest slice
  {
    const shard_permute_t p(18);
    PENTAGO_ASSERT_EQ(p.forward(0), 0u);
    PENTAGO_ASSERT_EQ(p.forward(1000000000000), 17933937948919u);
    PENTAGO_ASSERT_EQ(p.forward(65007135675647), 64869696720127u);
    PENTAGO_ASSERT_EQ(p.inverse(17933937948919), 1000000000000u);
  }
}

}  // namespace
}  // namespace pentago
