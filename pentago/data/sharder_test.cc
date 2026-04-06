// Sharder integration test: run the sharder on small data, verify output

#include "pentago/base/all_boards.h"
#include "pentago/base/superscore.h"
#include "pentago/data/arithmetic.h"
#include "pentago/data/shard.h"
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
  run(tfm::format("pentago/data/sharder --max-slice %d --shards %d data %s",
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
  run(tfm::format("pentago/data/sharder --max-slice %d --shards %d --range :8 --memory 0.00005 data %s",
                   max_slice, total_shards, tmp.path));
  run(tfm::format("pentago/data/sharder --max-slice %d --shards %d --range 8: --memory 0.00005 data %s",
                   max_slice, total_shards, tmp.path));
  verify_shards(max_slice, total_shards, range(total_shards), tmp.path);
}

// Boards sampled from shard 7 (slices 0-5, 32 total shards) and verified against
// the pentago server. Values: 1=current player wins, 0=tie, -1=current player loses.
//
// These depend on the shard permutation (shard_permute.h) and shard layout. Regenerate by:
//   1. In the correctness test below, after the ASSERT_EQ(actual, expected) line,
//      temporarily add a loop to dump ~100 boards:
//        Random rng(uint128_t(0xfeedface));
//        for (const auto& bv : actual)
//          if (rng.uniform<double>() < 100.0 / actual.size())
//            fprintf(stderr, "BOARD:%llu\n", (unsigned long long)bv.board);
//   2. Run the test to collect board IDs from stderr
//   3. Look up each board on the server:
//        curl https://us-central1-naml-148801.cloudfunctions.net/pentago/{board_id}
//      The board's own value is at key "{board_id}" in the JSON response.
//   4. Replace the map below with the new {board_id, value} pairs.
static const unordered_map<board_t, int> server_values = {{1723,0}, {2863,-1}, {13170,1}, {147111,0}, {592908,0}, {1839868,1}, {2560296,-1}, {3542025,0}, {7278924,1}, {10619051,-1}, {10624137,-1}, {17708091,0}, {31916053,1}, {37161110,0}, {79639443,1}, {95558238,1}, {100860688,0}, {159253995,0}, {287255952,0}, {433525338,0}, {525533217,1}, {859970361,-1}, {4353360361,0}, {8623751249,-1}, {12934460228,-1}, {13760803659,-1}, {25770199911,0}, {25844121609,0}, {26258178291,0}, {38654773145,0}, {38659227657,0}, {39085867014,1}, {64521832587,0}, {77331103744,0}, {116110393398,-1}, {347893532818,0}, {347894710812,0}, {347898052627,0}, {347908276365,1}, {347988557986,0}, {347989869030,0}, {348020349345,0}, {348328165376,1}, {348752447627,1}, {348864380928,1}, {349038977580,0}, {353057898496,1}, {360787869831,-1}, {695795318893,1}, {1043820382419,-1}, {1048370151424,1}, {1159642546194,0}, {3131082604547,0}, {3131127693393,1}, {3132034843353,0}, {3135469846582,1}, {3156811776081,0}, {3156849197056,1}, {3169781547009,1}, {3169983135987,1}, {5218444247040,1}, {9393173102592,0}, {9398248407337,0}, {9405979559427,0}, {9741018857472,1}, {12524126011446,0}, {18786792112857,0}, {28179567216779,1}, {28179568263169,1}, {28179710804373,0}, {28180204486656,0}, {28411210825728,1}, {28527300771840,1}, {28875065458688,1}, {31310322204933,0}, {46965515288657,0}, {46965642559497,0}, {56358571475683,0}, {56359325859840,0}, {57402269827072,1}, {844772823072840,-1}, {845121144946691,-1}, {7599940340613768,0}, {7599940430860647,0}, {22799627732517026,-1}, {205195953810309390,0}, {205197345379123203,-1}, {205204651147395090,1}, {205214044495740931,0}, {615586121963881560,0}};

// Shard iterator tests

class shard_iterator_test : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    init_threads(-1, -1);
    dir_.reset(new tempdir_t("iter"));
    run(tfm::format("pentago/data/sharder --max-slice %d --shards %d data %s",
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
    PENTAGO_ASSERT_EQ(a.board, b.board);
    PENTAGO_ASSERT_EQ(a.value, b.value);
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
    ASSERT_EQ(b->board, board);
    PENTAGO_ASSERT_EQ(shard_to_server_value(board, b->value), val);
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
    PENTAGO_ASSERT_EQ(a.board, batch[i].board);
    PENTAGO_ASSERT_EQ(a.value, batch[i].value);
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
