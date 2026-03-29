// Pseudorandom shard mapping for Pentago positions
//
// Maps every (section, position, rotation) tuple to a unique index in [0, total),
// shuffled via a fixed pseudorandom permutation. This scatters the 256 rotation
// values of each position across the index space.
#pragma once

#include "pentago/base/board.h"
#include "pentago/base/section.h"
#include "pentago/base/symmetry.h"
#include "pentago/utility/array.h"
#include "pentago/utility/range.h"
#include <unordered_map>
namespace pentago {

using std::unordered_map;

struct shard_mapping_t {
  const int max_slice;
  const Array<const section_t> sections;  // all sections ordered by (slice, index within slice)
  const unordered_map<section_t, int> section_id;  // inverse: section → index
  const Array<const uint64_t> offsets;  // prefix sum of section.size()*256

  shard_mapping_t(const int max_slice);
  ~shard_mapping_t();

  uint64_t total() const { return offsets.back(); }

  // Forward: (section, position index, rotation) → shuffled index in [0, total)
  uint64_t forward(const section_t section, const Vector<int,4> index,
                   const local_symmetry_t rotation) const;

  // Inverse: shuffled index → location
  struct location_t {
    section_t section;
    Vector<int,4> index;
    local_symmetry_t rotation;
  };
  location_t inverse(const uint64_t shuffled) const;

  // Compute the board_t for a shuffled index
  board_t board(const uint64_t shuffled) const;

  // Shard assignment: shuffled index → shard number in [0, shards)
  int shard(const int shards, const uint64_t shuffled) const;

  // Range of shuffled indices belonging to a given shard
  Range<uint64_t> shard_range(const int shards, const int shard) const;
};

}  // namespace pentago
