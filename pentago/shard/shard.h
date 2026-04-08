// Pseudorandom shard mapping for Pentago positions
//
// Maps every (section, position, rotation) tuple to a unique index in [0, total),
// shuffled via a fixed pseudorandom permutation. This scatters the 256 rotation
// values of each position across the index space.
#pragma once

#include "pentago/base/board.h"
#include "pentago/base/section.h"
#include "pentago/base/superscore.h"
#include "pentago/base/symmetry.h"
#include "pentago/shard/arithmetic.h"
#include "pentago/data/file.h"
#include "pentago/shard/shard_permute.h"
#include "pentago/shard/ternary.h"
#include "pentago/utility/array.h"
#include "pentago/utility/noncopyable.h"
#include "pentago/utility/random.h"
#include "pentago/utility/range.h"
#include <optional>
#include <unordered_map>
namespace pentago {

using std::optional;
using std::unordered_map;

struct shard_mapping_t {
  const int slice;
  const Array<const section_t> sections;  // sections for this slice only
  const unordered_map<section_t, int> section_id;  // inverse: section → index
  const Array<const uint64_t> offsets;  // prefix sum of section.size()*256
  const shard_permute_t permute;  // cached permutation for this slice

  shard_mapping_t(const int slice);
  ~shard_mapping_t();

  uint64_t total() const { return offsets.back(); }

  // Lookup section → index in sections array (asserts on missing)
  int section_index(const section_t section) const;

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
};

// Precomputed shard locator for power-of-two round-robin assignment.
// Maps shuffled indices to (shard, position) using bitmask and shift.
// Assert-free hot path; power-of-two check is in the constructor.
struct shard_locator_t {
  const int shard_mask;   // total_shards - 1
  const int shard_shift;  // __builtin_ctz(total_shards)
  const Range<int> shard_range;

  shard_locator_t(const int total_shards, const Range<int> shard_range)
    : shard_mask(total_shards - 1),
      shard_shift(__builtin_ctz(total_shards)),
      shard_range(shard_range) {
    GEODE_ASSERT(total_shards > 0 && (total_shards & (total_shards - 1)) == 0);
  }

  // Shard number for one shuffled value
  int shard(const uint64_t shuffled) const {
    return int(shuffled & shard_mask);
  }

  // Position within shard
  uint64_t position(const uint64_t shuffled) const {
    return shuffled >> shard_shift;
  }

  // Number of entries in shard `shard` given `total` entries overall
  uint64_t shard_size(const uint64_t total, const int shard) const {
    return (total + shard_mask - shard) >> shard_shift;
  }

  // Inverse: (shard, position within shard) → shuffled index
  uint64_t shuffled_index(const int shard, const uint64_t position) const {
    return (position << shard_shift) | shard;
  }

#if PENTAGO_SSE
  // 8-bit mask with interleaved bit order: {v0[0], v1[0], v0[1], v1[1], ...}.
  // Iterate with: for (auto m = mask; m; m.advance()) { int j = m.index(); ... }
  struct shard_mask_t {
    int bits;
    explicit operator bool() const { return bits; }
    // Deinterleave: bit b maps to index (b>>1) | ((b&1)<<2)
    int index() const { const int b = __builtin_ctz(bits); return (b >> 1) | ((b & 1) << 2); }
    void advance() { bits &= bits - 1; }
  };

  // Returns mask of which shuffled values land in [shard_range.lo, shard_range.hi).
  // Exact: no false positives or negatives. Returns 0 in the common case (no hits).
  shard_mask_t in_range8(const uint64x8 shuffled) const {
    const __m256i mask_v = _mm256_set1_epi64x(shard_mask);
    // Merge shard numbers into 8 x uint32 via shift+or (avoids expensive permutevar)
    const __m256i s0 = _mm256_and_si256(shuffled.v0, mask_v);
    const __m256i s1 = _mm256_and_si256(shuffled.v1, mask_v);
    const __m256i s8 = _mm256_or_si256(s0, _mm256_slli_epi64(s1, 32));
    // Exact range check: shard in [shard_range.lo, shard_range.hi)
    const __m256i ge_lo = _mm256_cmpgt_epi32(s8, _mm256_set1_epi32(shard_range.lo - 1));
    const __m256i lt_hi = _mm256_cmpgt_epi32(_mm256_set1_epi32(shard_range.hi), s8);
    return {_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_and_si256(ge_lo, lt_hi)))};
  }
#endif  // PENTAGO_SSE
};

// Shard file header
struct shard_header_t {
  static constexpr int magic_size = 20;
  static constexpr char magic[21] = "pentago shard      \n";
  static constexpr int header_size = 20 + 4 + 4 + 4 + 4;  // 36 bytes

  Vector<char,20> magic_bytes;
  uint32_t version;       // = 2 (v1 used modular Feistel permutation, v2 uses L/H bit-level)
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

// Shard filename: shard-NNNNN-of-NNNNN.pentago.shard (width adapts to total_shards)
string shard_filename(const int shards, const int shard);

struct board_value_t {
  board_t board;
  int value;  // 0/1/2: black_wins + 2 * white_wins
};

static inline bool operator==(const board_value_t& a, const board_value_t& b) {
  return a.board == b.board && a.value == b.value;
}

static inline bool operator<(const board_value_t& a, const board_value_t& b) {
  return a.board < b.board || (a.board == b.board && a.value < b.value);
}

struct shard_iterator_t : noncopyable_t {
  shard_iterator_t(const string& dir, const int total_shards, const Range<int> shard_range,
                   const uint128_t seed);
  ~shard_iterator_t();

  board_value_t next();
  void next_batch(RawArray<board_value_t> batch);

private:
  const string dir;
  const int total_shards;
  const Range<int> shard_range;
  const shard_locator_t locator;
  Random random;

  // Shard ordering
  int shard_cursor;
  uint128_t epoch_key;

  // Per-slice state
  struct slice_t {
    shard_mapping_t mapping;
    optional<ternaries_t> decoded;
    uint64_t remaining = 0;
    uint64_t cursor = 0;
    uint64_t shard_range_lo = 0;
    explicit slice_t(const int slice) : mapping(slice) {}
  };
  vector<slice_t> slices;
  uint64_t total_remaining;

  void load_next_shard();
};

// Convert a shard value (0=tie, 1=black wins, 2=white wins) to the server value
// (1=current player wins, 0=tie, -1=current player loses).
// Black is the absolute first player; whose turn it is follows stone count parity.
int shard_to_server_value(board_t board, int shard_value);

// Scatter one block's entries into shard ternary buffers.
// For each position in the block, computes all 256 rotation-shuffled indices,
// filters to shard_range, and atomically sets the corresponding ternary values.
void scatter_block(
    const shard_mapping_t& mapping,
    const int total_shards,
    const Range<int> shard_range,
    RawArray<ternaries_t> buffers,
    const section_t section, const int block_size,
    const Vector<uint8_t,4> block,
    RawArray<const Vector<super_t,2>,4> data);

}  // namespace pentago
