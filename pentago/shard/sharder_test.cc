// Sharder integration test: run the sharder on small data, verify output

#include "pentago/base/all_boards.h"
#include "pentago/base/superscore.h"
#include "pentago/shard/arithmetic.h"
#include "pentago/shard/shard.h"
#include "pentago/data/supertensor.h"
#include "pentago/utility/exceptions.h"
#include "pentago/utility/log.h"
#include "pentago/utility/range.h"
#include "pentago/utility/temporary.h"
#include "pentago/utility/test_assert.h"
#include "pentago/utility/thread.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <unordered_map>
namespace pentago {
namespace {

using std::sort;
using std::unordered_map;

static void run(const string& cmd) {
  slog(cmd);
  fflush(stdout);
  const int status = system(cmd.c_str());
  if (status)
    throw OSError(tfm::format("command '%s' failed with status %d", cmd, status));
}

// Verify that shard files in output_dir match supertensor data for the given shard range
static void verify_shards(const int max_slice, const int total_shards,
                          const Range<int> shard_range, const string& output_dir) {
  for (const int slice : range(max_slice + 1)) {
    const shard_mapping_t mapping(slice);

    // Open supertensors and build lookup
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

    // Check each shard in the range
    for (const int s : shard_range) {
      const auto path = tfm::format("%s/%s", output_dir, shard_filename(total_shards, s));
      const shard_file_t sf(path);
      PENTAGO_ASSERT_EQ(sf.header.max_slice, uint32_t(max_slice));
      PENTAGO_ASSERT_EQ(sf.header.shard_id, uint32_t(s));
      PENTAGO_ASSERT_EQ(sf.header.total_shards, uint32_t(total_shards));

      const auto group = sf.read_group(slice);
      const shard_locator_t locator(total_shards, shard_range);
      const uint64_t shard_sz = locator.shard_size(mapping.total(), s);
      PENTAGO_ASSERT_EQ(group.total(), shard_sz);
      const auto decoded = arithmetic_decode(group);
      PENTAGO_ASSERT_EQ(decoded.size, shard_sz);

      for (const uint64_t i : range(shard_sz)) {
        const auto loc = mapping.inverse(locator.shuffled_index(s, i));
        const uint64_t flat = index64(loc.section.shape(), loc.index);
        const auto& entry = super_data.at(loc.section).at(flat);
        const int expected = entry[0](loc.rotation.local) + 2 * entry[1](loc.rotation.local);
        PENTAGO_ASSERT_EQ(decoded[i], expected);
      }
    }
    slog("slice %d: verified %llu entries across %d shards", slice, mapping.total(),
         shard_range.size());
  }
}

// Full run: all shards, single batch
TEST(sharder, roundtrip) {
  init_threads(-1, -1);
  const int max_slice = 5;
  const int total_shards = 32;

  tempdir_t tmp("sharder");
  run(tfm::format("pentago/shard/sharder --max-slice %d --shards %d data %s",
                   max_slice, total_shards, tmp.path));
  verify_shards(max_slice, total_shards, range(total_shards), tmp.path);
}

// Split range + tiny memory to force multiple batches
TEST(sharder, batched_range) {
  init_threads(-1, -1);
  const int max_slice = 4;
  const int total_shards = 16;

  tempdir_t tmp("sharder-batched");

  // Run two halves with --range, and --memory small enough to force >1 batch
  run(tfm::format("pentago/shard/sharder --max-slice %d --shards %d --range :8 --memory 0.00005 data %s",
                   max_slice, total_shards, tmp.path));
  run(tfm::format("pentago/shard/sharder --max-slice %d --shards %d --range 8: --memory 0.00005 data %s",
                   max_slice, total_shards, tmp.path));
  verify_shards(max_slice, total_shards, range(total_shards), tmp.path);
}

// Dry-run: run with --dry-run=0.5 and verify output is strictly smaller than full run
TEST(sharder, dry_run) {
  init_threads(-1, -1);
  const int max_slice = 5;
  const int total_shards = 32;

  // Full run
  tempdir_t full("sharder-full");
  run(tfm::format("pentago/shard/sharder --max-slice %d --shards %d data %s",
                   max_slice, total_shards, full.path));

  // Dry run at 50%
  tempdir_t dry("sharder-dry");
  run(tfm::format("pentago/shard/sharder --max-slice %d --shards %d --dry-run 0.5 data %s",
                   max_slice, total_shards, dry.path));

  // Compare: dry-run should write fewer shards and less data
  int full_files = 0, dry_files = 0;
  uint64_t full_total = 0, dry_total = 0;
  for (const int s : range(total_shards)) {
    const auto name = shard_filename(total_shards, s);
    const auto full_path = tfm::format("%s/%s", full.path, name);
    const auto dry_path = tfm::format("%s/%s", dry.path, name);
    // Full run should have all files
    const auto full_sf = shard_file_t(full_path);
    full_files++;
    for (const int sl : range(max_slice + 1))
      full_total += full_sf.read_group(sl).total();
    // Dry run may not have this file
    if (FILE* f = fopen(dry_path.c_str(), "r")) {
      fclose(f);
      const auto dry_sf = shard_file_t(dry_path);
      dry_files++;
      for (const int sl : range(max_slice + 1))
        dry_total += dry_sf.read_group(sl).total();
    }
  }
  slog("full: %d files, %llu encoded; dry: %d files, %llu encoded (%.0f%% files, %.0f%% data)",
       full_files, full_total, dry_files, dry_total,
       double(dry_files) / full_files * 100, double(dry_total) / full_total * 100);
  ASSERT_LT(dry_files, full_files) << "dry-run should write fewer shard files";
  ASSERT_LT(dry_total, full_total) << "dry-run should produce less encoded data";
  ASSERT_GT(dry_files, 0) << "dry-run should write at least one shard file";
}

// Boards sampled from shard 7 (slices 0-5, 32 total shards) and verified against
// the pentago server. Values: 1=current player wins, 0=tie, -1=current player loses.
//
// These depend on the shard permutation (shard_permute.h) and shard layout.
// IMPORTANT: When these values change, bump the shard file format version in shard.h/shard.cc.
// Regenerate by:
//   1. In the correctness test below, after the ASSERT_EQ(actual, expected) line,
//      temporarily add a loop to dump ~100 boards:
//        Random rng(uint128_t(0xfeedface));
//        for (const auto& bv : actual)
//          if (rng.uniform<double>() < 100.0 / actual.size())
//            fprintf(stderr, "BOARD:%llu\n", (unsigned long long)bv.board());
//   2. Run the test to collect board IDs from stderr
//   3. Look up each board on the server:
//        curl https://us-central1-naml-148801.cloudfunctions.net/pentago/{board_id}
//      The board's own value is at key "{board_id}" in the JSON response.
//   4. Replace the map below with the new {board_id, value} pairs.
static const unordered_map<board_t, int> server_values = {{4540,1},{13387,0},{138551,-1},{594298,0},{1970465,-1},{3539136,1},{3546021,0},{7080579,0},{10623891,0},{18297672,0},{32055870,0},{47782469,1},{95754659,1},{143331654,0},{286654962,1},{429984134,1},{477771129,1},{860166582,-1},{1003297356,0},{4598726899,-1},{9163571200,-1},{12985892864,1},{13793034321,-1},{22336569345,-1},{25819486625,0},{26212237312,0},{38654771697,0},{38660027463,-1},{39615529203,0},{77320034724,1},{77486948352,0},{116840071186,0},{347897663937,0},{347903098970,-1},{348042559488,1},{348051607848,-1},{348131950592,1},{348752447630,-1},{348895648179,-1},{356538908672,-1},{360787869795,-1},{373688696832,1},{386587754496,-1},{695816553429,0},{1047982650421,0},{1082621952081,-1},{3135336153088,-1},{3135612846242,1},{3169700085760,1},{3169972525479,0},{3247043575808,1},{3247122678003,-1},{3478924493286,1},{3478974824610,1},{6262922477649,1},{9393427972258,0},{9406073929890,1},{9406985207808,0},{10088926085849,0},{12524984729681,-1},{18790497982881,-1},{28179443220642,0},{28179471532536,1},{28179588317670,0},{28180140464551,0},{28295534936064,1},{28411670691840,0},{28527443509248,0},{29261707739622,-1},{37572405758253,-1},{37572469457301,-1},{37585364975616,-1},{56358592774225,1},{56397266878467,-1},{844450758328321,0},{2552060977742796,0},{2552061078208593,0},{22799550424156299,1},{205195953807753459,0},{205197345376569726,0},{205204651131483432,0},{205214044352741377,0}};

// Shard iterator tests

class shard_iterator_test : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    init_threads(-1, -1);
    dir_.reset(new tempdir_t("iter"));
    run(tfm::format("pentago/shard/sharder --max-slice %d --shards %d data %s",
                     max_slice, total_shards, dir_->path));
  }
  static void TearDownTestSuite() {
    dir_.reset();
  }
  static constexpr int max_slice = 5;
  static constexpr int total_shards = 32;
  static unique_ptr<tempdir_t> dir_;
};
unique_ptr<tempdir_t> shard_iterator_test::dir_;

TEST_F(shard_iterator_test, determinism) {
  const uint128_t seed = 42;
  shard_iterator_t it1(dir_->path, total_shards, range(0, 3), seed);
  shard_iterator_t it2(dir_->path, total_shards, range(0, 3), seed);
  for (const int i : range(200)) {
    (void)i;
    const auto a = it1.next();
    const auto b = it2.next();
    PENTAGO_ASSERT_EQ(a.board(), b.board());
    PENTAGO_ASSERT_EQ(a.value(), b.value());
  }
}

TEST_F(shard_iterator_test, correctness) {
  const int shard_id = 7;
  const uint128_t seed = 123;

  // Build expected (board, value) pairs from direct shard decoding
  const auto path = tfm::format("%s/%s", dir_->path, shard_filename(total_shards, shard_id));
  const shard_file_t sf(path);
  vector<board_value_t> expected;
  for (const int s : range(max_slice + 1)) {
    const shard_mapping_t mapping(s);
    const shard_locator_t loc(total_shards, range(shard_id, shard_id + 1));
    const uint64_t shard_sz = loc.shard_size(mapping.total(), shard_id);
    const auto decoded = arithmetic_decode(sf.read_group(s));
    for (const uint64_t i : range(shard_sz))
      expected.push_back({mapping.board(loc.shuffled_index(shard_id, i)), decoded[i]});
  }
  sort(expected.begin(), expected.end());

  // Verify iterator produces the same set
  shard_iterator_t it(dir_->path, total_shards, range(shard_id, shard_id + 1), seed);
  vector<board_value_t> actual;
  actual.reserve(expected.size());
  for (const uint64_t i : range(uint64_t(expected.size()))) {
    (void)i;
    actual.push_back(it.next());
  }
  sort(actual.begin(), actual.end());
  ASSERT_EQ(actual, expected);

  // Check all server-verified boards appear in actual with correct values
  for (const auto& [board, val] : server_values) {
    const auto b = lower_bound(actual.begin(), actual.end(), board_value_t{board, 0});
    ASSERT_NE(b, actual.end());
    ASSERT_EQ(b->board(), board);
    PENTAGO_ASSERT_EQ(shard_to_server_value(board, b->value()), val);
  }
  slog("checked %d server-verified boards", int(server_values.size()));
}

TEST_F(shard_iterator_test, batch_equal) {
  const uint128_t seed = 456;
  const auto sr = range(0, 2);
  const int n = 53;

  shard_iterator_t it1(dir_->path, total_shards, sr, seed);
  shard_iterator_t it2(dir_->path, total_shards, sr, seed);
  const Array<board_value_t> batch(n, uninit);
  it2.next_batch(batch);
  for (const int i : range(n)) {
    const auto a = it1.next();
    PENTAGO_ASSERT_EQ(a.board(), batch[i].board());
    PENTAGO_ASSERT_EQ(a.value(), batch[i].value());
  }
}

TEST_F(shard_iterator_test, epoch) {
  const auto sr = range(3, 5);  // 2 shards
  const uint128_t seed = 789;
  shard_iterator_t it(dir_->path, total_shards, sr, seed);

  // Count total entries across all shards in range
  uint64_t total_entries = 0;
  for (const int shard_id : sr)
    for (const int s : range(max_slice + 1)) {
      const shard_mapping_t mapping(s);
      total_entries += shard_locator_t(total_shards, range(shard_id, shard_id + 1))
          .shard_size(mapping.total(), shard_id);
    }

  // Collect both epochs
  vector<board_value_t> epoch1, epoch2;
  epoch1.reserve(total_entries);
  epoch2.reserve(total_entries);
  for (const uint64_t i : range(total_entries)) { (void)i; epoch1.push_back(it.next()); }
  for (const uint64_t i : range(total_entries)) { (void)i; epoch2.push_back(it.next()); }

  // Different order (overwhelmingly likely with random mixing)
  ASSERT_NE(epoch1, epoch2);

  // Same entries
  sort(epoch1.begin(), epoch1.end());
  sort(epoch2.begin(), epoch2.end());
  ASSERT_EQ(epoch1, epoch2);
}

}  // namespace
}  // namespace pentago
