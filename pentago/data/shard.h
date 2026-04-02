// Pseudorandom shard mapping for Pentago positions
//
// Maps every (section, position, rotation) tuple to a unique index in [0, total),
// shuffled via a fixed pseudorandom permutation. This scatters the 256 rotation
// values of each position across the index space.
#pragma once

#include "pentago/base/board.h"
#include "pentago/base/section.h"
#include "pentago/base/symmetry.h"
#include "pentago/data/arithmetic.h"
#include "pentago/data/file.h"
#include "pentago/utility/array.h"
#include "pentago/utility/range.h"
#include <unordered_map>
namespace pentago {

using std::unordered_map;

struct shard_mapping_t {
  const int slice;
  const Array<const section_t> sections;  // sections for this slice only
  const unordered_map<section_t, int> section_id;  // inverse: section → index
  const Array<const uint64_t> offsets;  // prefix sum of section.size()*256

  shard_mapping_t(const int slice);
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

// Shard file header
struct shard_header_t {
  static constexpr int magic_size = 20;
  static constexpr char magic[21] = "pentago shard      \n";
  static constexpr int header_size = 20 + 4 + 4 + 4 + 4;  // 36 bytes

  Vector<char,20> magic_bytes;
  uint32_t version;       // = 1
  uint32_t max_slice;     // e.g. 18
  uint32_t shard_id;      // this shard's index [0, total_shards)
  uint32_t total_shards;  // total number of shards

  shard_header_t();  // fills in magic_bytes and version
  void pack(RawArray<uint8_t> buffer) const;
  static shard_header_t unpack(RawArray<const uint8_t> buffer);
};

// Write a complete shard file
void write_shard(const string& path, const shard_header_t& header,
                 RawArray<const arithmetic_t> groups);

// Read a shard file
struct shard_file_t {
  shard_header_t header;
  Array<const uint32_t> group_offsets;  // max_slice + 2 entries
  shared_ptr<const read_file_t> fd;

  shard_file_t(const string& path);
  ~shard_file_t();

  // Read and decode one slice's group
  arithmetic_t read_group(const int slice) const;
};

}  // namespace pentago
