// Integration test: run the sharder with GCS input/output and verify results.
//
// Usage:
//   gcs_sharder_test --credentials key.json --input gs://bucket/input --output gs://bucket/output
//
// The input bucket must contain slice-{0..N}.pentago supertensor files.
// The output path will be written to (and read back for verification).
// Uses small slices (0-5) and few shards (32) for a quick test.

#include "pentago/gcs/gcs.h"
#include "pentago/shard/arithmetic.h"
#include "pentago/shard/shard.h"
#include "pentago/data/supertensor.h"
#include "pentago/utility/debug.h"
#include "pentago/utility/log.h"
#include "pentago/utility/random.h"
#include "pentago/utility/range.h"
#include "pentago/utility/temporary.h"
#include "pentago/utility/thread.h"
#include <algorithm>
#include <cstdlib>
#include <getopt.h>
#include <unordered_map>
namespace pentago {
namespace {

using std::sort;
using std::unordered_map;

struct options_t {
  string credentials;
  string input;
  string output;
  int max_slice = 5;
  int total_shards = 32;
};

static options_t parse_options(int argc, char** argv) {
  options_t o;
  static const struct option options[] = {
    {"credentials", required_argument, 0, 'c'},
    {"input", required_argument, 0, 'i'},
    {"output", required_argument, 0, 'o'},
    {"max-slice", required_argument, 0, 's'},
    {0, 0, 0, 0},
  };
  for (;;) {
    int c = getopt_long(argc, argv, "", options, nullptr);
    if (c == -1) break;
    switch (c) {
      case 'c': o.credentials = optarg; break;
      case 'i': o.input = optarg; break;
      case 'o': o.output = optarg; break;
      case 's': o.max_slice = atoi(optarg); break;
      default: die("unknown option");
    }
  }
  if (o.credentials.empty() || o.input.empty() || o.output.empty())
    die("usage: gcs_sharder_test --credentials KEY --input gs://... --output gs://...");
  return o;
}

static void run(const string& cmd) {
  slog(cmd);
  fflush(stdout);
  const int status = system(cmd.c_str());
  if (status) die("command failed with status %d: %s", status, cmd);
}

// Verify shard contents against supertensor data, reading both from GCS
static void verify_shards(const options_t& o) {
  for (const int slice : range(o.max_slice + 1)) {
    const shard_mapping_t mapping(slice);
    const auto readers = open_supertensors(tfm::format("data/slice-%d.pentago", slice));

    unordered_map<section_t, int> section_to_reader;
    for (const int ri : range(int(readers.size())))
      section_to_reader[readers[ri]->header.section] = ri;

    // Read all blocks into a map for lookup
    unordered_map<section_t, unordered_map<uint64_t, Vector<super_t,2>>> super_data;
    for (const auto& section : mapping.sections) {
      const auto& reader = *readers[section_to_reader.at(section)];
      const auto blocks = Vector<int,4>(reader.header.blocks);
      auto& section_map = super_data[section];
      for (const int b0 : range(blocks[0]))
        for (const int b1 : range(blocks[1]))
          for (const int b2 : range(blocks[2]))
            for (const int b3 : range(blocks[3])) {
              const auto block = Vector<uint8_t,4>(vec(b0, b1, b2, b3));
              const auto data = reader.read_block(block);
              const auto base = Vector<int,4>(block) * int(reader.header.block_size);
              const auto shape = data.shape();
              for (const int i0 : range(shape[0]))
                for (const int i1 : range(shape[1]))
                  for (const int i2 : range(shape[2]))
                    for (const int i3 : range(shape[3])) {
                      const auto index = base + vec(i0, i1, i2, i3);
                      const uint64_t flat = index64(section.shape(), index);
                      section_map[flat] = data(i0, i1, i2, i3);
                    }
            }
    }

    // Verify each shard
    const auto shard_range = range(o.total_shards);
    for (const int s : shard_range) {
      const auto path = tfm::format("%s/%s", o.output, shard_filename(o.total_shards, s));
      const shard_file_t sf(open_file(path));
      GEODE_ASSERT(sf.header.max_slice == uint32_t(o.max_slice));
      GEODE_ASSERT(sf.header.shard_id == uint32_t(s));
      GEODE_ASSERT(sf.header.total_shards == uint32_t(o.total_shards));

      const auto group = sf.read_group(slice);
      const shard_locator_t locator(o.total_shards, shard_range);
      const uint64_t shard_sz = locator.shard_size(mapping.total(), s);
      GEODE_ASSERT(group.total() == shard_sz);
      const auto decoded = arithmetic_decode(group);
      GEODE_ASSERT(uint64_t(decoded.size) == shard_sz);

      for (const uint64_t i : range(shard_sz)) {
        const auto loc = mapping.inverse(locator.shuffled_index(s, i));
        const uint64_t flat = index64(loc.section.shape(), loc.index);
        const auto& entry = super_data.at(loc.section).at(flat);
        const int expected = entry[0](loc.rotation.local) + 2 * entry[1](loc.rotation.local);
        GEODE_ASSERT(decoded[i] == expected);
      }
    }
    slog("slice %d: verified %llu entries across %d shards", slice, mapping.total(),
         o.total_shards);
  }
}

// Verify shard_iterator_t works by decoding a shard from GCS and checking
// that the iterator produces the same (board, value) pairs.
static void verify_iterator(const options_t& o) {
  const int shard_id = 7;
  const auto path = tfm::format("%s/%s", o.output, shard_filename(o.total_shards, shard_id));
  const shard_file_t sf(open_file(path));

  // Build expected (board, value) pairs from direct shard decoding
  vector<board_value_t> expected;
  for (const int s : range(o.max_slice + 1)) {
    const shard_mapping_t mapping(s);
    const shard_locator_t loc(o.total_shards, range(shard_id, shard_id + 1));
    const uint64_t shard_sz = loc.shard_size(mapping.total(), shard_id);
    const auto decoded = arithmetic_decode(sf.read_group(s));
    for (const uint64_t i : range(shard_sz))
      expected.push_back({mapping.board(loc.shuffled_index(shard_id, i)), decoded[i]});
  }
  sort(expected.begin(), expected.end());

  // Verify the iterator produces the same set. The iterator reads shard files
  // from local paths, so we download shard 7 to a temp file first.
  // (shard_iterator_t doesn't support GCS paths directly.)
  const auto shard_data = serialize_shard(sf.header, [&]() {
    Array<arithmetic_t> groups(o.max_slice + 1);
    for (const int s : range(o.max_slice + 1))
      groups[s] = sf.read_group(s);
    return groups;
  }());

  // Write to temp file for the iterator
  tempdir_t tmp("gcs-iter");
  const auto local_path = tfm::format("%s/%s", tmp.path, shard_filename(o.total_shards, shard_id));
  const auto fd = write_local_file(local_path);
  const auto err = fd->pwrite(shard_data, 0);
  GEODE_ASSERT(err.empty(), err);

  const uint128_t seed = 123;
  shard_iterator_t it(tmp.path, o.total_shards, range(shard_id, shard_id + 1), seed);
  vector<board_value_t> actual;
  actual.reserve(expected.size());
  for (const uint64_t i __attribute__((unused)) : range(uint64_t(expected.size())))
    actual.push_back(it.next());
  sort(actual.begin(), actual.end());
  GEODE_ASSERT(actual.size() == expected.size());
  for (const uint64_t i : range(uint64_t(expected.size()))) {
    GEODE_ASSERT(actual[i].board == expected[i].board);
    GEODE_ASSERT(actual[i].value == expected[i].value);
  }
  slog("iterator: verified %llu entries from shard %d", uint64_t(expected.size()), shard_id);
}

void toplevel(int argc, char** argv) {
  const auto o = parse_options(argc, argv);
  gcs_init(o.credentials);
  init_threads(-1, -1);

  const string sharder = "pentago/shard/sharder";

  // Run the sharder with GCS paths
  slog("=== Running sharder ===");
  run(tfm::format("%s --max-slice %d --shards %d --credentials %s %s %s",
                   sharder, o.max_slice, o.total_shards, o.credentials, o.input, o.output));

  // Verify output shards against supertensor data
  slog("=== Verifying shards ===");
  verify_shards(o);

  // Verify iterator matches decoded shard contents
  slog("=== Verifying iterator ===");
  verify_iterator(o);

  slog("PASSED: all checks passed");
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
