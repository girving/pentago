// Microbenchmark for scatter_block and random_permute

#include "pentago/data/shard.h"
#include "pentago/data/ternary.h"
#include "pentago/base/all_boards.h"
#include "pentago/base/superscore.h"
#include "pentago/utility/log.h"
#include "pentago/utility/permute.h"
#include "pentago/utility/range.h"
#include "pentago/utility/wall_time.h"
#include "pentago/utility/test_assert.h"
#include "gtest/gtest.h"
namespace pentago {
namespace {

// Min-of-N timing for stable results
static constexpr int timing_iterations = 10;

// Benchmark random_permute in isolation: 256 consecutive calls (one position's worth)
TEST(shard_bench, random_permute) {
  const shard_mapping_t mapping(4);
  const uint64_t n = mapping.total();
  const uint128_t key = (uint128_t(0xb7e151628aed2a6a) << 64) | 0xbf7158809cf4f3c7;

  // Pick a base offset in the middle of the range
  const uint64_t base = n / 3;
  const int reps = 10000;

  double best = 1e18;
  uint64_t sink = 0;  // prevent optimization
  for (const int iter __attribute__((unused)) : range(timing_iterations)) {
    const auto start = wall_time();
    for (const int rep : range(reps)) {
      const uint64_t b = base + uint64_t(rep) * 256;
      for (const int r : range(256))
        sink += random_permute(n, key, b + r);
    }
    const double elapsed = (wall_time() - start).seconds();
    best = std::min(best, elapsed);
  }
  const double ns_per_call = best / (reps * 256) * 1e9;
  const double ns_per_position = best / reps * 1e9;
  slog("random_permute: %.1f ns/call, %.0f ns/position (256 calls), sink=%llu",
       ns_per_call, ns_per_position, sink);
}

// Benchmark scatter_block with a single shard at realistic shard counts
static void bench_scatter(const int slice) {
  const shard_mapping_t mapping(slice);
  const int total_shards = 100000;
  const int shard_id = total_shards / 2;
  const auto shard_range = range(shard_id, shard_id + 1);

  // Allocate single shard buffer
  const auto sr = mapping.shard_range(total_shards, shard_id);
  Array<uint64_t> shard_los(1, uninit);
  shard_los[0] = sr.lo;
  vector<ternaries_t> buffers;
  buffers.emplace_back(sr.size());

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
       slice, mapping.total(), sr.size(), positions);

  double best = 1e18;
  for (const int iter __attribute__((unused)) : range(timing_iterations)) {
    buffers[0] = ternaries_t(sr.size());
    const auto start = wall_time();
    scatter_block(mapping, total_shards, shard_range, shard_los,
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
