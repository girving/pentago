// Time the tiled midsolver on real boards, one raw board per line
//
// Usage: tiled_bench <boards.txt> [count] [threads] [merged] [check]
//   count:   number of boards to solve (default all)
//   threads: participants in the worker pool (default 1)
//   merged:  1 for one pass computing win and not-lose together (default 0)
//   check:   1 to compare every value against a dense midsolve (default 0)

#include "pentago/mid/tiled.h"
#include "pentago/mid/codec.h"
#include "pentago/mid/midengine.h"
#include "pentago/utility/log.h"
#include "pentago/utility/memory.h"
#include "pentago/utility/wall_time.h"
#include <algorithm>
#include <fstream>
#include <unordered_map>
namespace pentago {
namespace {

using std::sort;
using std::unordered_map;

void bench(const char* path, const int count, const int threads, const bool merged, const bool check) {
  vector<raw_t> raws;
  std::ifstream file(path);
  for (raw_t raw; file >> raw && int(raws.size()) < count;)
    raws.push_back(raw);
  GEODE_ASSERT(raws.size());
  tiled_options_t opts;
  opts.threads = threads;
  opts.merged = merged;
  vector<double> times;
  int64_t peak = 0;
  for (const raw_t raw : raws) {
    const auto board = high_board_t::from_raw(raw);
    tiled_stats_t stats;
    reset_codec_counts();
    const auto start = wall_time();
    const auto values = midsolve_tiled(board, opts, &stats);
    const double t = (wall_time() - start).seconds();
    times.push_back(t);
    peak = std::max(peak, stats.peak_bytes);
    slog("board %llu, %d stones: %.2f s (decode %.2f, compute %.2f), peak %.1f MB, arena %.1f MB, values %d",
         (unsigned long long)raw, board.count(), t, stats.decode_seconds, stats.compute_seconds,
         stats.peak_bytes / double(1 << 20), stats.arena_bytes / double(1 << 20), int(values.size()));
    const auto counts = codec_counts();
    uint64_t total = 0;
    for (const auto m : counts.modes) total += m;
    slog("  codec: %.2f bits/entry; all %.3f, mtf %.3f, xor %.3f, prev %.3f, far %.3f, sparse %.3f, dmtf %.3f, raw %.3f",
         counts.bits / double(total), counts.modes[0] / double(total), counts.modes[1] / double(total),
         counts.modes[2] / double(total), counts.modes[3] / double(total), counts.modes[4] / double(total),
         counts.modes[5] / double(total), counts.modes[6] / double(total), counts.modes[7] / double(total));
    if (check) {
      unordered_map<raw_t,int> dense;
      for (const auto& [b, v] : midsolve(board, midsolve_workspace(board.count())))
        dense[b] = v;
      for (const auto& [b, v] : values)
        GEODE_ASSERT(dense.at(b) == v);
      slog("  check passed (%d values)", int(values.size()));
    }
  }
  sort(times.begin(), times.end());
  slog("threads %d, merged %d: %d boards, median %.2f s, max %.2f s, peak arena %.1f MB, process %s",
       threads, merged, int(times.size()), times[times.size() / 2], times.back(), peak / double(1 << 20),
       memory_report(memory_info()));
}

}  // namespace
}  // namespace pentago

int main(int argc, char** argv) {
  using namespace pentago;
  GEODE_ASSERT(2 <= argc && argc <= 6);
  bench(argv[1], argc > 2 ? atoi(argv[2]) : 1 << 30, argc > 3 ? atoi(argv[3]) : 1,
        argc > 4 && atoi(argv[4]), argc > 5 && atoi(argv[5]));
  return 0;
}
