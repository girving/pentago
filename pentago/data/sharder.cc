// Convert supertensor files to shard files
//
// Reads .pentago supertensor files and produces .pentago.shard files.
// Each shard contains rANS-coded ternary outcomes (win/tie/loss) for
// a contiguous range of shuffled positions, with per-slice coding.

#include "pentago/base/all_boards.h"
#include "pentago/data/arithmetic.h"
#include "pentago/data/shard.h"
#include "pentago/data/supertensor.h"
#include "pentago/data/ternary.h"
#include "pentago/utility/debug.h"
#include "pentago/utility/log.h"
#include "pentago/utility/range.h"
#include "pentago/utility/thread.h"
#include <atomic>
#include <cstdlib>
#include <optional>
#include <cstring>
#include <thread>
#include <vector>
#include <getopt.h>
#include "pentago/utility/memory.h"
namespace pentago {
namespace {

using std::nullopt;
using std::optional;
using std::atomic;
using std::thread;

struct options_t {
  string input_dir;
  string output_dir;
  int max_slice = 18;
  int total_shards = 100000;
  string shard_range = ":";  // Python-style lo:hi
  int threads = -1;   // -1 means auto-detect
  int64_t memory = -1;  // -1 means 80% of system RAM
};

static options_t parse_options(int argc, char** argv) {
  options_t o;
  static const option options[] = {
      {"help", no_argument, 0, 'h'},
      {"max-slice", required_argument, 0, 's'},
      {"shards", required_argument, 0, 'n'},
      {"range", required_argument, 0, 'r'},
      {"threads", required_argument, 0, 't'},
      {"memory", required_argument, 0, 'm'},
      {0, 0, 0, 0},
  };
  for (;;) {
    int option = 0;
    int c = getopt_long(argc, argv, "", options, &option);
    if (c == -1) break;
    switch (c) {
      case 'h':
        slog("usage: sharder [options] <input_dir> <output_dir>");
        slog("Convert supertensor files to shard files.");
        slog("  -h, --help              Display usage information and quit");
        slog("  --max-slice N           Maximum slice to process (default: 18)");
        slog("  --shards N              Total number of shards (default: 100000)");
        slog("  --range lo:hi           Shard range to generate (default: all)");
        slog("  --threads N             CPU worker threads (default: auto-detect)");
        slog("  --memory N              Memory budget in GB (default: 80%% of system RAM)");
        exit(0);
      case 's': o.max_slice = atoi(optarg); break;
      case 'n': o.total_shards = atoi(optarg); break;
      case 'r': o.shard_range = optarg; break;
      case 't': o.threads = atoi(optarg); break;
      case 'm': o.memory = int64_t(atof(optarg) * (1LL << 30)); break;
      default: die("impossible option character %d", c);
    }
  }
  const int nargs = argc - optind;
  if (nargs != 2)
    die("expected 2 arguments <input_dir> <output_dir>, got %d", nargs);
  o.input_dir = argv[optind];
  o.output_dir = argv[optind + 1];
  GEODE_ASSERT(0 <= o.max_slice && o.max_slice <= 18);
  GEODE_ASSERT(0 < o.total_shards);
  if (o.memory < 0)
    o.memory = int64_t(total_memory() * 80 / 100);
  return o;
}

struct progress_t {
  const char* label;
  const uint64_t total;
  atomic<uint64_t> done{0};
  const wall_time_t start = wall_time();

  progress_t(const char* label, const uint64_t total) : label(label), total(total) {}

  void tick() {
    const uint64_t d = done.fetch_add(1, std::memory_order_relaxed) + 1;
    if (d % 1000 == 0 || d == total) {
      const double elapsed = (wall_time() - start).seconds();
      const double eta = d < total ? elapsed * (double(total) / d - 1) : 0;
      slog("    %s: %llu / %llu, %.1fs elapsed, %.1fs remaining", label, d, total, elapsed, eta);
    }
  }
};

struct block_work_t {
  int reader_index;
  Vector<uint8_t,4> block;
};

template<class F>
static void parallel_for(const int num_threads, const size_t n, const F& f) {
  GEODE_ASSERT(num_threads > 0);
  vector<thread> threads;
  threads.reserve(num_threads);
  atomic<size_t> next(0);
  for (const int t __attribute__((unused)) : range(num_threads))
    threads.emplace_back([&]() {
      for (;;) {
        const size_t i = next.fetch_add(1, std::memory_order_relaxed);
        if (i >= n) break;
        f(i);
      }
    });
  for (auto& t : threads) t.join();
}

void toplevel(int argc, char** argv) {
  const auto o = parse_options(argc, argv);
  Scope scope("sharder");
  const int threads = o.threads >= 0 ? o.threads : default_threads();
  init_threads(1, 1);

  const auto shard_range = parse_range(o.shard_range, o.total_shards);
  const int target_shards = shard_range.size();
  slog("input: %s, output: %s", o.input_dir, o.output_dir);
  slog("max_slice: %d, shards: %d, range: [%d, %d)", o.max_slice, o.total_shards,
       shard_range.lo, shard_range.hi);
  slog("threads: %d, memory: %.1f GB", threads, double(o.memory) / (1LL << 30));

  // Compute batch size from the largest slice (max_slice), which dominates memory.
  // Peak memory per shard: raw ternary buffer for the current slice, plus compressed
  // data accumulated across all already-encoded slices (~half the raw size). Use 2x.
  const shard_mapping_t max_mapping(o.max_slice);
  const auto max_range = max_mapping.shard_range(o.total_shards, shard_range.lo);
  const uint64_t max_entries_per_shard = max_range.hi - max_range.lo;
  const uint64_t mem_per_shard = ((max_entries_per_shard + 4) / 5 + 64) * 2;
  const int shards_per_batch = std::max(1, !mem_per_shard ? target_shards :
      int(std::min(uint64_t(target_shards), uint64_t(o.memory) / mem_per_shard)));
  const int n_batches = (target_shards + shards_per_batch - 1) / shards_per_batch;
  slog("%llu max entries/shard, %d shards/batch, %d batches",
       max_entries_per_shard, shards_per_batch, n_batches);

  // Precompute per-slice data: mappings, readers, section lookup
  struct slice_info_t {
    shard_mapping_t mapping;
    vector<shared_ptr<const supertensor_reader_t>> readers;
    unordered_map<section_t, int> section_to_reader;
    uint32_t total_blocks;
  };
  vector<slice_info_t> slices;
  slices.reserve(o.max_slice + 1);
  for (const int slice : range(o.max_slice + 1)) {
    auto readers = open_supertensors(tfm::format("%s/slice-%d.pentago", o.input_dir, slice));
    unordered_map<section_t, int> section_to_reader;
    for (const int ri : range(int(readers.size())))
      section_to_reader[readers[ri]->header.section] = ri;
    uint32_t total_blocks = 0;
    for (const auto& reader : readers)
      total_blocks += Vector<uint32_t,4>(reader->header.blocks).product();
    slices.emplace_back(shard_mapping_t(slice), std::move(readers),
                        std::move(section_to_reader), total_blocks);
  }
  shutdown_threads();  // free spinning pool threads before the main computation

  // Process batches of shards
  for (const int bi : range(n_batches)) {
    const auto batch = partition_loop(target_shards, n_batches, bi);
    const int abs_lo = shard_range.lo + batch.lo;
    const int abs_hi = shard_range.lo + batch.hi;
    Scope batch_scope(tfm::format("batch %d/%d (shards %d..%d)",
                                   bi + 1, n_batches, abs_lo, abs_hi - 1));

    // Accumulated encoded groups for this batch, indexed [shard within batch][slice]
    vector<vector<arithmetic_t>> batch_groups(batch.size(), vector<arithmetic_t>(o.max_slice + 1));

    // Process slices from largest to smallest, since largest is slowest
    for (int slice = o.max_slice; slice >= 0; slice--) {
      Scope slice_scope(tfm::format("slice %d", slice));
      const auto& si = slices[slice];

      // Allocate ternary buffers for this batch and slice
      vector<optional<ternaries_t>> buffers;
      buffers.reserve(batch.size());
      for (const int b : range(batch.size()))
        buffers.emplace_back(si.mapping.shard_range(o.total_shards, abs_lo + b).size());

      // Precompute shard range starts for this batch
      Array<uint64_t> shard_los(batch.size(), uninit);
      for (const int b : range(batch.size()))
        shard_los[b] = si.mapping.shard_range(o.total_shards, abs_lo + b).lo;
      const auto shard_los_raw = shard_los.raw();

      // Scatter blocks into shard buffers using atomic adds
      progress_t block_progress("blocks", si.total_blocks);
      auto scatter_block = [&, shard_los_raw](const section_t section, const int block_size,
                               Vector<uint8_t,4> block, Array<Vector<super_t,2>,4> data) {
        const auto base_index = Vector<int,4>(block) * block_size;
        const auto block_shape = data.shape();
        for (const int i0 : range(block_shape[0]))
          for (const int i1 : range(block_shape[1]))
            for (const int i2 : range(block_shape[2]))
              for (const int i3 : range(block_shape[3])) {
                const auto index = base_index + vec(i0, i1, i2, i3);
                const auto& entry = data(i0, i1, i2, i3);
                const auto& black_wins = entry[0];
                const auto& white_wins = entry[1];
                for (const uint8_t r : range(256)) {
                  const auto rot = local_symmetry_t(r);
                  const uint64_t shuffled = si.mapping.forward(section, index, rot);
                  const int s = si.mapping.shard(o.total_shards, shuffled);
                  if (s < abs_lo || s >= abs_hi)
                    continue;
                  const uint64_t pos = shuffled - shard_los_raw[s - abs_lo];
                  buffers[s - abs_lo]->atomic_set_from_zero(pos, black_wins(r) + 2 * white_wins(r));
                }
              }
        block_progress.tick();
      };

      // Read and scatter blocks using parallel_for with natural backpressure
      vector<block_work_t> work;
      for (const auto& section : si.mapping.sections) {
        const int ri = si.section_to_reader.at(section);
        const auto& reader = *si.readers[ri];
        const auto blocks = Vector<int,4>(reader.header.blocks);
        for (const int b0 : range(blocks[0]))
          for (const int b1 : range(blocks[1]))
            for (const int b2 : range(blocks[2]))
              for (const int b3 : range(blocks[3]))
                work.emplace_back(ri, Vector<uint8_t,4>(vec(b0, b1, b2, b3)));
      }
      GEODE_ASSERT(work.size() == si.total_blocks);
      Random(slice).shuffle(work);
      parallel_for(threads, work.size(), [&](const size_t wi) {
        const auto& w = work[wi];
        const auto& reader = *si.readers[w.reader_index];
        scatter_block(reader.header.section, reader.header.block_size, w.block,
                      reader.read_block_sync(w.block));
      });

      // Encode each shard's ternary buffer in parallel, freeing as we go
      progress_t encode_progress("encode", batch.size());
      parallel_for(threads, batch.size(), [&](const size_t b) {
        batch_groups[b][slice] = arithmetic_encode(*buffers[b]);
        buffers[b] = nullopt;
        encode_progress.tick();
      });
    }  // slice

    // Write this batch's shard files in parallel
    progress_t write_progress("write", batch.size());
    parallel_for(threads, batch.size(), [&](const size_t b) {
      const int abs_shard = abs_lo + int(b);
      shard_header_t h;
      h.max_slice = o.max_slice;
      h.shard_id = abs_shard;
      h.total_shards = o.total_shards;
      const auto path = tfm::format("%s/%s", o.output_dir,
                                    shard_filename(o.total_shards, abs_shard));
      write_shard(path, h, asarray(batch_groups[b]));
      write_progress.tick();
    });
    slog("wrote %d shard files", batch.size());
  }  // batch

  slog("done: %d shard files written to %s", target_shards, o.output_dir);
}

}  // namespace
}  // namespace pentago

int main(int argc, char** argv) {
  try {
    pentago::toplevel(argc, argv);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
