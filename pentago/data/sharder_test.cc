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
      const auto shard_r = mapping.shard_range(total_shards, s);
      PENTAGO_ASSERT_EQ(group.total(), shard_r.size());
      const auto decoded = arithmetic_decode(group);
      PENTAGO_ASSERT_EQ(decoded.size, shard_r.size());

      for (const uint64_t i : range(shard_r.size())) {
        const auto loc = mapping.inverse(shard_r.lo + i);
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
  const int total_shards = 41;  // prime to avoid coincidences

  tempdir_t tmp("sharder");
  run(tfm::format("pentago/data/sharder --max-slice %d --shards %d data %s",
                   max_slice, total_shards, tmp.path));
  verify_shards(max_slice, total_shards, range(total_shards), tmp.path);
}

// Split range + tiny memory to force multiple batches
TEST(sharder, batched_range) {
  init_threads(-1, -1);
  const int max_slice = 4;
  const int total_shards = 17;

  tempdir_t tmp("sharder-batched");

  // Run two halves with --range, and --memory small enough to force >1 batch
  // Slice 4 has ~42K entries/shard → ~8.5 KB packed, so 0.00005 GB (~50 KB) fits ~5 shards
  run(tfm::format("pentago/data/sharder --max-slice %d --shards %d --range :9 --memory 0.00005 data %s",
                   max_slice, total_shards, tmp.path));
  run(tfm::format("pentago/data/sharder --max-slice %d --shards %d --range 9: --memory 0.00005 data %s",
                   max_slice, total_shards, tmp.path));
  verify_shards(max_slice, total_shards, range(total_shards), tmp.path);
}

// Boards sampled from shard 7 (slices 0-5, 41 total shards) and verified against
// the pentago server. Values: 1=current player wins, 0=tie, -1=current player loses.
//
// These depend on the shard permutation (shard_permute.h). If the permutation changes,
// regenerate by:
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
static const unordered_map<board_t, int> server_values = {{2465,1}, {4487,0}, {4638,1}, {65606,1}, {70658,0}, {466204,0}, {1771013,1}, {3605242,0}, {5310172,-1}, {5312822,-1}, {5315626,0}, {5316684,0}, {5322285,1}, {5323026,1}, {5323730,-1}, {17109306,0}, {26542648,-1}, {32048400,0}, {95622130,0}, {143394231,0}, {143987452,0}, {429995592,0}, {430112782,0}, {433525257,1}, {4295360546,0}, {4327211278,0}, {4470149425,0}, {4470341650,0}, {9307160577,0}, {12890218329,0}, {13061455872,1}, {13840613457,1}, {38702481426,-1}, {38808715426,0}, {51560251392,-1}, {115964510480,-1}, {115969438617,0}, {116190609408,0}, {116397047862,0}, {124570959872,-1}, {141735165961,0}, {155781365760,1}, {231960091797,0}, {232390066188,0}, {244983005184,0}, {347893924092,0}, {347895300125,-1}, {347908279923,0}, {347908290237,0}, {347924201493,1}, {348037841058,-1}, {348180905993,0}, {360904655091,-1}, {386644770816,0}, {695816814673,0}, {695880318985,1}, {696215863566,0}, {696358404177,0}, {696644671914,1}, {708823744512,0}, {1044537016419,1}, {1044547829760,1}, {1047985586176,-1}, {1048290525211,0}, {1404836511744,1}, {1739615764480,0}, {2087355351292,-1}, {2087370424348,-1}, {2091841355776,0}, {3131031882353,0}, {3131036467607,0}, {3131057713986,1}, {3131114520576,0}, {3140528832513,0}, {3156819050496,0}, {3170837790720,1}, {3208771928064,0}, {3826911477841,0}, {5218385723392,0}, {9405988995567,0}, {9509921095680,1}, {12524490915840,0}, {18786338537472,-1}, {28179668140032,1}, {28180140400470,-1}, {28180331495910,0}, {28187887469273,-1}, {28192175947938,1}, {28333909870002,-1}, {28527183855616,1}, {29227284312898,0}, {29261612318774,-1}, {31310312767506,1}, {31311031762944,1}, {31658215735296,-1}, {56706502230016,0}, {7600520166506740,0}, {22799477413839352,0}, {22799821149242820,-1}, {22800168930118014,-1}, {22800517077277575,0}, {205195606057750968,0}, {1846757335083647273,0}};

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
  static constexpr int total_shards = 41;
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
    const auto sr = mapping.shard_range(total_shards, shard_id);
    const auto decoded = arithmetic_decode(sf.read_group(s));
    for (const uint64_t i : range(sr.size()))
      expected.push_back({mapping.board(sr.lo + i), decoded[i]});
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
      total_entries += mapping.shard_range(total_shards, shard_id).size();
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
