// Convert supertensor files to shard files
//
// Reads .pentago supertensor files and produces .pentago.shard files.
// Each shard contains rANS-coded ternary outcomes (win/tie/loss) for
// a contiguous range of shuffled positions, with per-slice coding.

#include "pentago/base/all_boards.h"
#include "pentago/gcs/gcs.h"
#include "pentago/gcs/stream.h"
#include "pentago/shard/arithmetic.h"
#include "pentago/shard/shard.h"
#include "pentago/shard/ternary.h"
#include "pentago/utility/debug.h"
#include "pentago/utility/log.h"
#include "pentago/shard/parallel.h"
#include "pentago/utility/range.h"
#include "pentago/utility/thread.h"
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <getopt.h>
#include "pentago/utility/memory.h"
namespace pentago {
namespace {

using std::atomic;

struct options_t {
  string input_dir;
  string output_dir;
  int max_slice = 18;
  int total_shards = 100000;
  string shard_range = ":";  // Python-style lo:hi
  int threads = -1;   // -1 means auto-detect
  int64_t memory = -1;  // -1 means 80% of system RAM
  double dry_run = 1.0;  // fraction of work to do (1.0 = full run)
  string credentials;    // GCS service account JSON key file (required for gs:// paths)
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
      {"dry-run", required_argument, 0, 'd'},
      {"credentials", required_argument, 0, 'c'},
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
        slog("  --dry-run F             Do fraction F of work, e.g. 0.01 for 1%% (default: 1.0)");
        slog("  --credentials FILE      GCS service account JSON key file (required for gs:// paths)");
        exit(0);
      case 's': o.max_slice = atoi(optarg); break;
      case 'n': o.total_shards = atoi(optarg); break;
      case 'r': o.shard_range = optarg; break;
      case 't': o.threads = atoi(optarg); break;
      case 'm': o.memory = int64_t(atof(optarg) * (1LL << 30)); break;
      case 'd': o.dry_run = atof(optarg); break;
      case 'c': o.credentials = optarg; break;
      default: die("impossible option character %d", c);
    }
  }
  const int nargs = argc - optind;
  if (nargs != 2)
    die("expected 2 arguments <input_dir> <output_dir>, got %d", nargs);
  o.input_dir = argv[optind];
  o.output_dir = argv[optind + 1];
  GEODE_ASSERT(0 <= o.max_slice && o.max_slice <= 18);
  GEODE_ASSERT(o.total_shards > 0 && (o.total_shards & (o.total_shards - 1)) == 0);
  GEODE_ASSERT(o.dry_run > 0 && o.dry_run <= 1.0);
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
      slog("%s: %llu / %llu, %.1fs elapsed, %.1fs remaining", label, d, total, elapsed, eta);
      if (d % 100000 == 0)
        slog("memory: %s", memory_report(memory_info()));
    }
  }
};

void toplevel(int argc, char** argv) {
  const auto o = parse_options(argc, argv);
  if (is_gcs_path(o.input_dir) || is_gcs_path(o.output_dir)) {
    if (o.credentials.empty()) die("gs:// paths require --credentials");
    gcs_init(o.credentials);
  }
  const auto dry = [&](const size_t n) { return std::max(size_t(1), size_t(n * o.dry_run)); };
  Scope scope("sharder");
  const int threads = o.threads >= 0 ? o.threads : default_threads();
  init_threads(1, 1);

  const auto shard_range = parse_range(o.shard_range, o.total_shards);
  const int target_shards = shard_range.size();
  slog("input: %s, output: %s", o.input_dir, o.output_dir);
  slog("max_slice: %d, shards: %d, range: [%d, %d)", o.max_slice, o.total_shards,
       shard_range.lo, shard_range.hi);
  slog("threads: %d, memory: %.1f GB%s", threads, double(o.memory) / (1LL << 30),
       o.dry_run < 1.0 ? tfm::format(", dry-run: %.2f%%", o.dry_run * 100).c_str() : "");

  // Compute batch size from the largest slice (max_slice), which dominates memory.
  // Peak memory per shard: raw ternary buffer for the current slice, plus compressed
  // data accumulated across all already-encoded slices (~half the raw size). Use 2x.
  const shard_mapping_t max_mapping(o.max_slice);
  const shard_locator_t locator(o.total_shards, shard_range);
  const uint64_t max_entries_per_shard = locator.shard_size(max_mapping.total(), shard_range.lo);
  const uint64_t mem_per_shard = ((max_entries_per_shard + 4) / 5 + 64) * 2;
  const int shards_per_batch = std::max(1, !mem_per_shard ? target_shards :
      int(std::min(uint64_t(target_shards), uint64_t(o.memory) / mem_per_shard)));
  const int n_batches = (target_shards + shards_per_batch - 1) / shards_per_batch;
  slog("%llu max entries/shard, %d shards/batch, %d batches",
       max_entries_per_shard, shards_per_batch, n_batches);

  // Precompute per-slice shard mappings
  vector<shard_mapping_t> mappings;
  mappings.reserve(o.max_slice + 1);
  for (const int slice : range(o.max_slice + 1))
    mappings.emplace_back(slice);
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
      const auto& mapping = mappings[slice];

      // Allocate ternary buffers for this batch and slice
      vector<ternaries_t> buffers;
      buffers.reserve(batch.size());
      for (const int b : range(batch.size()))
        buffers.emplace_back(locator.shard_size(mapping.total(), abs_lo + b));
      const auto abs_range = range(abs_lo, abs_hi);

      // Stream blocks with multi-threaded readahead (works for both local and GCS)
      const auto input_path = tfm::format("%s/slice-%d.pentago", o.input_dir, slice);
      supertensor_stream_t stream(input_path, 10LL << 30, 8);
      const auto dry_blocks = dry(stream.total_blocks());
      progress_t block_progress("blocks", dry_blocks);
      parallel_for(threads, dry_blocks, [&](const size_t) {
        auto block = stream.next();
        if (!block) return;
        const auto data = supertensor_stream_t::decompress(block);
        scatter_block(mapping, o.total_shards, abs_range,
                      buffers, block.section,
                      block.block, data);
        block_progress.tick();
      });
      stream.check_reader_error();

      // Encode each shard's ternary buffer in parallel, freeing as we go
      const auto dry_encode = dry(batch.size());
      progress_t encode_progress("encode", dry_encode);
      parallel_for(threads, dry_encode, [&](const size_t b) {
        batch_groups[b][slice] = arithmetic_encode(buffers[b]);
        buffers[b] = ternaries_t();
        encode_progress.tick();
      });
    }  // slice

    // Write this batch's shard files in parallel
    const auto dry_write = dry(batch.size());
    progress_t write_progress("write", dry_write);
    parallel_for(threads, dry_write, [&](const size_t b) {
      const int abs_shard = abs_lo + int(b);
      shard_header_t h;
      h.max_slice = o.max_slice;
      h.shard_id = abs_shard;
      h.total_shards = o.total_shards;
      const auto path = tfm::format("%s/%s", o.output_dir,
                                    shard_filename(o.total_shards, abs_shard));
      if (is_gcs_path(path)) {
        const auto buf = serialize_shard(h, asarray(batch_groups[b]));
        gcs_upload(path, buf);
      } else {
        write_shard(path, h, asarray(batch_groups[b]));
      }
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
