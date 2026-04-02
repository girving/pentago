// Shard mapping and file format tests

#include "pentago/data/shard.h"
#include "pentago/data/arithmetic.h"
#include "pentago/base/all_boards.h"
#include "pentago/base/symmetry.h"
#include "pentago/utility/range.h"
#include "pentago/utility/log.h"
#include "pentago/utility/temporary.h"
#include "pentago/utility/test_assert.h"
#include "gtest/gtest.h"
#include <numeric>
#include <unordered_set>
namespace pentago {
namespace {

using std::unordered_set;

TEST(shard, total) {
  for (const int slice : range(4 + 1)) {
    const shard_mapping_t m(slice);
    uint64_t expected = 0;
    for (const auto& s : all_boards_sections(slice, 8))
      expected += s.size() * 256;
    PENTAGO_ASSERT_EQ(m.total(), expected);
    slog("slice %d: %d sections, %llu total entries", slice, m.sections.size(), m.total());
  }
}

TEST(shard, exhaustive_roundtrip) {
  for (const int slice : range(4 + 1)) {
    const shard_mapping_t m(slice);
    unordered_set<uint64_t> seen;
    for (const int si : range(m.sections.size())) {
      const auto& section = m.sections[si];
      const auto shape = section.shape();
      for (const int i0 : range(shape[0]))
        for (const int i1 : range(shape[1]))
          for (const int i2 : range(shape[2]))
            for (const int i3 : range(shape[3]))
              for (const int r : range(256)) {
                const auto index = vec(i0, i1, i2, i3);
                const auto rot = local_symmetry_t(uint8_t(r));
                const uint64_t shuffled = m.forward(section, index, rot);
                ASSERT_LT(shuffled, m.total());
                ASSERT_TRUE(seen.insert(shuffled).second)
                    << "collision at slice " << slice << " section " << section
                    << " index " << index << " rot " << int(rot.local);
                const auto loc = m.inverse(shuffled);
                ASSERT_EQ(loc.section, section);
                ASSERT_EQ(loc.index, index);
                PENTAGO_ASSERT_EQ(loc.rotation.local, rot.local);
              }
    }
    PENTAGO_ASSERT_EQ(seen.size(), m.total());
  }
}

TEST(shard, board_roundtrip) {
  for (const int slice : range(4 + 1)) {
    const shard_mapping_t m(slice);
    for (const uint64_t i : range(m.total())) {
      const auto b = m.board(i);
      const auto section = count(b);
      ASSERT_TRUE(section.valid());
      PENTAGO_ASSERT_EQ(section.sum(), slice);
    }
  }
}

TEST(shard, shard_distribution) {
  for (const int slice : range(4 + 1)) {
    const shard_mapping_t m(slice);
    const int shards = 17;  // prime to avoid coincidences
    // Verify shard_range partitions [0, total) exactly
    uint64_t prev_hi = 0;
    for (const int s : range(shards)) {
      const auto r = m.shard_range(shards, s);
      PENTAGO_ASSERT_EQ(r.lo, prev_hi);
      ASSERT_LE(r.lo, r.hi);
      prev_hi = r.hi;
    }
    PENTAGO_ASSERT_EQ(prev_hi, m.total());
    // Verify shard() is consistent with shard_range() for every index
    for (const uint64_t i : range(m.total())) {
      const int s = m.shard(shards, i);
      const auto r = m.shard_range(shards, s);
      ASSERT_LE(r.lo, i);
      ASSERT_LT(i, r.hi);
    }
  }
}

TEST(shard, header_roundtrip) {
  shard_header_t h;
  h.max_slice = 18;
  h.shard_id = 42;
  h.total_shards = 100000;

  Array<uint8_t> buf(shard_header_t::header_size, uninit);
  h.pack(buf);
  const auto h2 = shard_header_t::unpack(buf);
  PENTAGO_ASSERT_EQ(h2.version, 1u);
  PENTAGO_ASSERT_EQ(h2.max_slice, 18u);
  PENTAGO_ASSERT_EQ(h2.shard_id, 42u);
  PENTAGO_ASSERT_EQ(h2.total_shards, 100000u);
}

TEST(shard, file_roundtrip) {
  const int max_slice = 3;
  const int n_groups = max_slice + 1;

  // Create and encode groups with different sizes
  vector<arithmetic_t> groups;
  for (const int s : range(n_groups)) {
    const uint64_t n = 100 + s * 137;  // different sizes per slice
    ternaries_t data(n);
    for (uint64_t i = 0; i < n; i++)
      data.set(i, int((i * 11 + s * 3) % 3));
    groups.push_back(arithmetic_encode(data));
  }

  // Build header
  shard_header_t h;
  h.max_slice = max_slice;
  h.shard_id = 5;
  h.total_shards = 13;

  // Write
  tempdir_t tmp("shard");
  const string path = tmp.path + "/test.pentago.shard";
  write_shard(path, h, asarray(groups));

  // Read back
  const shard_file_t sf(path);
  PENTAGO_ASSERT_EQ(sf.header.version, 1u);
  PENTAGO_ASSERT_EQ(sf.header.max_slice, uint32_t(max_slice));
  PENTAGO_ASSERT_EQ(sf.header.shard_id, 5u);
  PENTAGO_ASSERT_EQ(sf.header.total_shards, 13u);

  // Verify each group roundtrips
  for (const int s : range(n_groups)) {
    const auto g = sf.read_group(s);
    const uint64_t n = 100 + s * 137;
    PENTAGO_ASSERT_EQ(g.total(), n);
    const auto decoded = arithmetic_decode(g);
    PENTAGO_ASSERT_EQ(decoded.size, n);
    for (uint64_t i = 0; i < n; i++)
      PENTAGO_ASSERT_EQ(decoded[i], int((i * 11 + s * 3) % 3));
  }
}

}  // namespace
}  // namespace pentago
