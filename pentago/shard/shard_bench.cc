// Microbenchmark for scatter_block and shard_permute
//
// Speed log (Cascade Lake 2.5 GHz, -c opt -march=native, min of 10 iterations):
// "position" = one board position = 256 ternary values (one per rotation).
//
//   shard_permute (slice 18): 1.8 ns/call, 467 ns/position (~4.5 cycles/value)
//
//   scatter_block (131072 shards, 1 shard in range, 4096 positions/block):
//     slice 18: 569 ns/position
//
//   Before shift+or shard merge (permutevar pack in in_range8):
//     scatter_block slice 18 was ~630 ns/position
//
//   forward4 (single __m256i, 4 values) was tried but is 10-15% slower than
//   forward8 (two __m256i, 8 values): OoO engine overlaps the two independent
//   dependency chains, filling pipeline bubbles. Measured 2.0 ns/call for forward4
//   vs 1.8 ns/call for forward8.
//
//   Before L/H bit-level permutation (modular Feistel with 32x32->64 multiplies):
//     shard_permute was 6.5 ns/call, 1670 ns/position (~16 cycles/value)
//     scatter_block was ~1640 ns/position
//
//   Before round-robin (100000 shards, double-reciprocal locator):
//     scatter_block was ~6600 ns/position (4x slower)
//   Before Feistel permutation (random_permute via hash):
//     scatter_block was ~23000 ns/position (3.5x slower than double-reciprocal)

#include "pentago/shard/shard.h"
#include "pentago/shard/shard_permute.h"
#include "pentago/shard/ternary.h"
#include "pentago/base/all_boards.h"
#include "pentago/base/superscore.h"
#include "pentago/utility/log.h"
#include "pentago/utility/range.h"
#include "pentago/utility/wall_time.h"
#include "pentago/utility/test_assert.h"
#include "gtest/gtest.h"
namespace pentago {
namespace {

// Min-of-N timing for stable results
static constexpr int timing_iterations = 100;

// Benchmark shard_permute in isolation: 256 consecutive calls (one position's worth)
TEST(shard_bench, shard_permute) {
  const shard_permute_t perm(18);

  // Pick a base offset in the middle of the range
  const uint64_t base = perm.n / 3;
  const int reps = 10000;

  double best = 1e18;
  uint64_t sink = 0;  // prevent optimization
  for (const int iter __attribute__((unused)) : range(timing_iterations)) {
    const auto start = wall_time();
    for (const int rep : range(reps)) {
      const uint64_t b = base + uint64_t(rep) * 256;
#if PENTAGO_SSE
      const __m256i off0 = _mm256_setr_epi64x(0, 1, 2, 3);
      const __m256i off1 = _mm256_setr_epi64x(4, 5, 6, 7);
      for (int r = 0; r < 256; r += 8) {
        const __m256i bv = _mm256_set1_epi64x(b + r);
        const auto y = perm.forward8({_mm256_add_epi64(bv, off0), _mm256_add_epi64(bv, off1)});
        // Extract one lane to prevent dead code elimination
        sink += _mm_cvtsi128_si64(_mm256_castsi256_si128(y.v0));
      }
#else
      for (const int r : range(256))
        sink += perm.forward(b + r);
#endif
    }
    const double elapsed = (wall_time() - start).seconds();
    best = std::min(best, elapsed);
  }
  const double ns_per_call = best / (reps * 256) * 1e9;
  const double ns_per_position = best / reps * 1e9;
  slog("shard_permute: %.1f ns/call, %.0f ns/position (256 calls), sink=%llu",
       ns_per_call, ns_per_position, sink);
}

// Benchmark scatter_block with a single shard at realistic shard counts.
// Buffer is allocated and touched once before timing to eliminate memset
// and page fault noise from the measurement.
static void bench_scatter(const int slice) {
  const shard_mapping_t mapping(slice);
  const int total_shards = 131072;  // power of two
  const int shard_id = total_shards / 2;
  const auto shard_range = range(shard_id, shard_id + 1);

  const shard_locator_t locator(total_shards, shard_range);
  vector<ternaries_t> buffers;
  buffers.emplace_back(locator.shard_size(mapping.total(), shard_id));

  // Pick a representative section and create a synthetic block
  const auto& section = mapping.sections[mapping.sections.size() / 2];
  const auto shape = section.shape();
  const int block_size = 8;
  const auto block = Vector<uint8_t,4>();
  const auto block_shape = vec(std::min(block_size, shape[0]), std::min(block_size, shape[1]),
                               std::min(block_size, shape[2]), std::min(block_size, shape[3]));
  const int positions = block_shape.product();
  Array<Vector<super_t,2>,4> data(block_shape);  // zero-initialized (all ties)

  slog("slice %d: %llu total, %llu entries/shard, %d positions/block",
       slice, mapping.total(), locator.shard_size(mapping.total(), shard_id), positions);

  // Don't re-zero between iterations: violates atomic_set_from_zero's precondition,
  // but we're only timing the permute + shard lookup path, not the writes.
  double best = 1e18;
  for (const int iter __attribute__((unused)) : range(timing_iterations)) {
    const auto start = wall_time();
    scatter_block(mapping, total_shards, shard_range,
                  buffers, section, block_size, block, data);
    const double elapsed = (wall_time() - start).seconds();
    best = std::min(best, elapsed);
  }
  const double ns_per_pos = best / positions * 1e9;
  slog("  %.0f ns/position, %.3f ms total", ns_per_pos, best * 1e3);
}

TEST(shard_bench, scatter_16) { bench_scatter(16); }
TEST(shard_bench, scatter_17) { bench_scatter(17); }
TEST(shard_bench, scatter_18) { bench_scatter(18); }

}  // namespace
}  // namespace pentago
