// Pseudorandom shard mapping for Pentago positions

#include "pentago/data/shard.h"
#include "pentago/base/all_boards.h"
#include "pentago/utility/const_cast.h"
#include "pentago/utility/debug.h"
#include "pentago/utility/index.h"
#include "pentago/utility/permute.h"
#include "pentago/utility/range.h"
#include <algorithm>
namespace pentago {

using std::get;
using std::upper_bound;

// Fixed key for the pseudorandom permutation (digits of e in hex)
static const uint128_t shard_key = (uint128_t(0xb7e151628aed2a6a) << 64) | 0xbf7158809cf4f3c7;

shard_mapping_t::shard_mapping_t(const int max_slice)
    : max_slice(max_slice) {
  GEODE_ASSERT(max_slice <= 18);
  // Collect all sections ordered by (slice, index within slice)
  vector<section_t> all;
  for (const int n : range(max_slice + 1))
    for (const auto& s : all_boards_sections(n, 8))
      all.push_back(s);
  const_cast_(sections) = asarray(all).copy();

  // Build inverse map and prefix sums
  Array<uint64_t> off(sections.size() + 1, uninit);
  off[0] = 0;
  for (const int i : range(sections.size())) {
    const_cast_(section_id)[sections[i]] = i;
    off[i + 1] = off[i] + sections[i].size() * 256;
  }
  const_cast_(offsets) = off;
}

shard_mapping_t::~shard_mapping_t() {}

uint64_t shard_mapping_t::forward(const section_t section, const Vector<int,4> index,
                                  const local_symmetry_t rotation) const {
  const auto it = section_id.find(section);
  GEODE_ASSERT(it != section_id.end());
  const auto shape = section.shape();
  GEODE_ASSERT(valid(shape, index));
  const uint64_t pos = index64(shape, index);
  const uint64_t linear = offsets[it->second] + pos * 256 + rotation.local;
  return random_permute(total(), shard_key, linear);
}

shard_mapping_t::location_t shard_mapping_t::inverse(const uint64_t shuffled) const {
  GEODE_ASSERT(shuffled < total());
  const uint64_t linear = random_unpermute(total(), shard_key, shuffled);

  // Binary search to find section
  const int si = int(upper_bound(offsets.begin(), offsets.end(), linear) - offsets.begin()) - 1;
  GEODE_ASSERT(unsigned(si) < unsigned(sections.size()));

  // Decompose within section
  const uint64_t within = linear - offsets[si];
  const uint64_t pos = within / 256;
  const int rot = int(within % 256);
  const auto shape = sections[si].shape();
  return {sections[si], decompose64(shape, pos), local_symmetry_t(uint8_t(rot))};
}

board_t shard_mapping_t::board(const uint64_t shuffled) const {
  const auto loc = inverse(shuffled);
  quadrant_t quads[4];
  for (const int i : range(4)) {
    const auto rmin = get<0>(rotation_minimal_quadrants(loc.section.counts[i]));
    quads[i] = rmin[loc.index[i]];
  }
  const auto b = quadrants(quads[0], quads[1], quads[2], quads[3]);
  return transform_board(symmetry_t(loc.rotation), b);
}

int shard_mapping_t::shard(const int shards, const uint64_t shuffled) const {
  GEODE_ASSERT(0 < shards && shards <= 100000);  // Prevent overflow
  return int(shuffled * shards / total());
}

Range<uint64_t> shard_mapping_t::shard_range(const int shards, const int shard) const {
  GEODE_ASSERT(0 < shards && shards <= 100000);  // Prevent overflow
  GEODE_ASSERT(unsigned(shard) < unsigned(shards));
  const uint64_t lo = total() * shard / shards;
  const uint64_t hi = total() * (shard + 1) / shards;
  return {lo, hi};
}

}  // namespace pentago
