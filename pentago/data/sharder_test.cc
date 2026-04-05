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
// https://us-central1-naml-148801.cloudfunctions.net/pentago.
// Values: 1=current player wins, 0=tie, -1=current player loses.
static const unordered_map<board_t, int> server_values = {{95562693,0}, {734873124864,0}, {2533622693377563,0}, {28192208388096,1}, {22818259300649102,0}, {17112348,0}, {22828696166598066,0}, {2087863713792,1}, {31851282,1}, {32440842,0}, {38655177539,0}, {28179296361029,0}, {9015,0}, {1774638,0}, {18902438314065,0}, {143918532,1}, {695928422427,1}, {3733,0}, {347989803009,0}, {17212115083,-1}, {68399115136008273,1}, {10696219,1}, {38667092131,0}, {6266797883392,1}, {3131050623465,0}, {77886265761,0}, {95560983,-1}, {38672400408,0}, {196411572,-1}, {431755544,1}, {348179660962,-1}, {7599867321253894,0}, {3139796336640,0}, {26058489856,-1}, {48955888,0}, {116072054865,0}, {1846757335179071847,0}, {695833663905,1}, {22799705195741185,0}, {1275609415680,1}, {28179376439296,1}, {594630,0}, {28179333526392,0}, {3259912032534,0}, {17710515,0}, {91060174848,0}, {347902968222,0}, {347897660285,-1}, {201722,0}, {22799550518722803,-1}, {1069591364313,-1}, {143526675,0}, {348035678391,0}, {11208854,0}, {2533300655753598,0}, {6262094430208,1}, {3547697,-1}, {4726325410,0}, {13171562932,0}, {696788058841,0}, {21312324,-1}, {29235853000866,-1}, {6378173300745,-1}, {22799821435909092,-1}, {116059671153,1}, {4736752033,0}, {2087842480857,0}, {49549645,0}, {9393131225250,0}, {1043708903590,0}, {3140528832593,0}, {4325244928,-1}, {3208680308736,0}, {3363867197440,0}, {440611896,0}, {348183134289,0}, {3131031750072,1}, {9509090623515,1}, {3479236706304,1}, {53085393,0}, {15925830,1}, {5298257938,1}, {29222992871424,1}, {3169781809152,1}, {12388502,0}, {9393095704738,0}, {28205624721408,-1}, {347940127269,0}, {193369671819,0}, {116060850316,-1}, {28295961182214,0}, {48435532,0}, {2091649531985,0}, {116118061542,1}, {844773109143975,0}, {5310704,0}, {695788503121,0}, {19134223810803,-1}, {1159647658146,0}, {37172278,0}, {5384893,0}, {124605367001,0}, {6262110554251,0}, {22818259332366340,0}, {1159644250112,1}, {348332953881,0}};

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
