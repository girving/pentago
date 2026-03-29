// Shard mapping tests

#include "pentago/data/shard.h"
#include "pentago/base/all_boards.h"
#include "pentago/base/symmetry.h"
#include "pentago/utility/range.h"
#include "pentago/utility/log.h"
#include "pentago/utility/test_assert.h"
#include "gtest/gtest.h"
#include <numeric>
#include <unordered_set>
namespace pentago {
namespace {

using std::unordered_set;

TEST(shard, total) {
  for (const int max_slice : range(4 + 1)) {
    const shard_mapping_t m(max_slice);
    uint64_t expected = 0;
    for (const int n : range(max_slice + 1))
      for (const auto& s : all_boards_sections(n, 8))
        expected += s.size() * 256;
    PENTAGO_ASSERT_EQ(m.total(), expected);
    slog("max_slice %d: %d sections, %llu total entries", max_slice, m.sections.size(), m.total());
  }
}

TEST(shard, exhaustive_roundtrip) {
  for (const int max_slice : range(4 + 1)) {
    const shard_mapping_t m(max_slice);
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
                    << "collision at max_slice " << max_slice << " section " << section
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
  for (const int max_slice : range(4 + 1)) {
    const shard_mapping_t m(max_slice);
    for (const uint64_t i : range(m.total())) {
      const auto b = m.board(i);
      const auto section = count(b);
      ASSERT_TRUE(section.valid());
      PENTAGO_ASSERT_LE(section.sum(), m.max_slice);
    }
  }
}

TEST(shard, shard_distribution) {
  for (const int max_slice : range(4 + 1)) {
    const shard_mapping_t m(max_slice);
    const int shards = 16;
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

}  // namespace
}  // namespace pentago
