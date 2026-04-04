// Pseudorandom shard mapping for Pentago positions

#include "pentago/data/shard.h"
#include "pentago/base/all_boards.h"
#include "pentago/utility/char_view.h"
#include "pentago/utility/ceil_div.h"
#include "pentago/utility/const_cast.h"
#include "pentago/utility/debug.h"
#include "pentago/utility/endian.h"
#include "pentago/utility/index.h"
#include "pentago/utility/permute.h"
#include "pentago/utility/range.h"
#include <algorithm>
#include <cstring>
namespace pentago {

using std::get;
using std::make_unique;
using std::memcpy;
using std::upper_bound;

// Fixed key for the pseudorandom permutation (digits of e in hex)
static const uint128_t shard_key = (uint128_t(0xb7e151628aed2a6a) << 64) | 0xbf7158809cf4f3c7;

shard_mapping_t::shard_mapping_t(const int slice)
    : slice(slice) {
  GEODE_ASSERT(0 <= slice && slice <= 18);
  const_cast_(sections) = asarray(all_boards_sections(slice, 8)).copy();

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
  // shuffled < total(), and total() * shards must fit in uint64_t (checked in shard_range)
  return int(shuffled * shards / total());
}

Range<uint64_t> shard_mapping_t::shard_range(const int shards, const int shard) const {
  GEODE_ASSERT(0 < shards && shards <= 100000);  // Prevent overflow
  GEODE_ASSERT(unsigned(shard) < unsigned(shards));
  // Use ceiling division to be the exact inverse of shard():
  // shard(i) = floor(i*shards/T), so shard s owns i where i*shards >= s*T, i.e. i >= ceil(s*T/shards).
  const uint64_t T = total();
  // T * shards must fit in uint64_t
  GEODE_ASSERT(T <= numeric_limits<uint64_t>::max() / shards);
  const uint64_t lo = ceil_div(T * shard, uint64_t(shards));
  const uint64_t hi = ceil_div(T * (shard + 1), uint64_t(shards));
  return {lo, hi};
}

shard_header_t::shard_header_t()
    : version(1), max_slice(0), shard_id(0), total_shards(0) {
  memcpy(magic_bytes.data(), magic, magic_size);
}

#define SHARD_HEADER_FIELDS() \
  FIELD(magic_bytes) \
  FIELD(version) \
  FIELD(max_slice) \
  FIELD(shard_id) \
  FIELD(total_shards)

void shard_header_t::pack(RawArray<uint8_t> buffer) const {
  GEODE_ASSERT(buffer.size() == header_size);
  int next = 0;
  #define FIELD(f) ({ \
    GEODE_ASSERT(next+sizeof(f)<=size_t(header_size)); \
    const auto le = native_to_little_endian(f); \
    memcpy(buffer.data()+next,&le,sizeof(le)); \
    next += sizeof(le); });
  SHARD_HEADER_FIELDS()
  #undef FIELD
  GEODE_ASSERT(next == header_size);
}

shard_header_t shard_header_t::unpack(RawArray<const uint8_t> buffer) {
  GEODE_ASSERT(buffer.size() == header_size);
  GEODE_ASSERT(!memcmp(buffer.data(), magic, magic_size));
  shard_header_t h;
  int next = 0;
  #define FIELD(f) ({ \
    GEODE_ASSERT(next+sizeof(h.f)<=size_t(header_size)); \
    decltype(h.f) le; \
    memcpy(&le,buffer.data()+next,sizeof(le)); \
    h.f = little_to_native_endian(le); \
    next += sizeof(le); });
  SHARD_HEADER_FIELDS()
  #undef FIELD
  GEODE_ASSERT(next == header_size);
  GEODE_ASSERT(h.version == 1);
  GEODE_ASSERT(h.max_slice <= 18);
  GEODE_ASSERT(h.shard_id < h.total_shards);
  return h;
}

void write_shard(const string& path, const shard_header_t& header,
                 RawArray<const arithmetic_t> groups) {
  const int n = header.max_slice + 1;
  GEODE_ASSERT(groups.size() == n);

  // Compute offsets from serialized sizes
  uint64_t offset = shard_header_t::header_size + (n + 1) * 4;
  Array<uint32_t> offsets(n + 1, uninit);
  for (const int i : range(n)) {
    GEODE_ASSERT(offset <= numeric_limits<uint32_t>::max());
    offsets[i] = uint32_t(offset);
    offset += groups[i].serialized_size();
  }
  GEODE_ASSERT(offset <= numeric_limits<uint32_t>::max());
  offsets[n] = uint32_t(offset);

  // Write header and offset table
  const auto fd = write_local_file(path);
  Array<uint8_t> hbuf(shard_header_t::header_size, uninit);
  header.pack(hbuf);
  auto err = fd->pwrite(hbuf, 0);
  GEODE_ASSERT(err.empty(), err);
  to_little_endian_inplace(offsets);
  err = fd->pwrite(char_view(offsets), shard_header_t::header_size);
  GEODE_ASSERT(err.empty(), err);

  // Serialize and write each group
  for (const int i : range(n)) {
    const auto buf = arithmetic_serialize(groups[i]);
    err = fd->pwrite(buf, little_to_native_endian(offsets[i]));
    GEODE_ASSERT(err.empty(), err);
  }
}

shard_file_t::shard_file_t(const string& path)
    : fd(read_local_file(path)) {
  // Read header
  Array<uint8_t> hbuf(shard_header_t::header_size, uninit);
  auto err = fd->pread(hbuf, 0);
  GEODE_ASSERT(err.empty(), err);
  header = shard_header_t::unpack(hbuf);

  // Read offset table
  const int n = header.max_slice + 1;
  Array<uint32_t> offs(n + 1, uninit);
  err = fd->pread(char_view(offs), shard_header_t::header_size);
  GEODE_ASSERT(err.empty(), err);
  to_little_endian_inplace(offs);
  group_offsets = offs;
}

shard_file_t::~shard_file_t() {}

arithmetic_t shard_file_t::read_group(const int slice) const {
  GEODE_ASSERT(0 <= slice && slice <= int(header.max_slice));
  const uint32_t start = group_offsets[slice];
  const uint32_t end = group_offsets[slice + 1];
  GEODE_ASSERT(start < end);
  Array<uint8_t> buf(int(end - start), uninit);
  const auto err = fd->pread(buf, start);
  GEODE_ASSERT(err.empty(), err);
  return arithmetic_deserialize(buf);
}

shard_iterator_t::shard_iterator_t(const string& dir, const int total_shards,
                                   const Range<int> shard_range, const uint128_t seed)
    : dir(dir), total_shards(total_shards), shard_range(shard_range), random(seed),
      shard_cursor(0), epoch_key(random.bits<uint128_t>()), total_remaining(0) {
  GEODE_ASSERT(!shard_range.empty());
  load_next_shard();
}

shard_iterator_t::~shard_iterator_t() {}

void shard_iterator_t::load_next_shard() {
  if (shard_cursor >= shard_range.size()) {
    epoch_key = random.bits<uint128_t>();
    shard_cursor = 0;
  }
  const int permuted = int(random_permute(uint64_t(shard_range.size()), epoch_key,
                                          uint64_t(shard_cursor++)));
  const int shard_id = shard_range.lo + permuted;
  const auto path = tfm::format("%s/shard-%05d-of-%05d.pentago.shard",
                                dir, shard_id, total_shards - 1);
  const shard_file_t sf(path);

  // Initialize slices lazily on first shard load
  if (slices.empty()) {
    const int n = sf.header.max_slice + 1;
    slices.reserve(n);
    for (const int s : range(n))
      slices.emplace_back(s);
  }
  GEODE_ASSERT(sf.header.max_slice + 1 == uint32_t(slices.size()));
  GEODE_ASSERT(sf.header.total_shards == uint32_t(total_shards));

  // Decode all groups and set up per-slice state
  total_remaining = 0;
  for (auto& slice : slices) {
    slice.decoded.emplace(arithmetic_decode(sf.read_group(slice.mapping.slice)));
    const auto sr = slice.mapping.shard_range(total_shards, shard_id);
    slice.remaining = sr.size();
    slice.cursor = 0;
    slice.shard_range_lo = sr.lo;
    total_remaining += sr.size();
  }
}

board_value_t shard_iterator_t::next() {
  if (total_remaining == 0)
    load_next_shard();

  // Weighted random selection of slice
  const int last = int(slices.size()) - 1;
  uint64_t r = random.uniform(total_remaining);
  int s = 0;
  for (; s < last; s++) {
    if (r < slices[s].remaining) break;
    r -= slices[s].remaining;
  }

  // Read entry and reconstruct board
  auto& slice = slices[s];
  const int value = (*slice.decoded)[slice.cursor];
  const board_t board = slice.mapping.board(slice.shard_range_lo + slice.cursor);

  // Advance
  slice.cursor++;
  slice.remaining--;
  total_remaining--;

  return {board, value};
}

void shard_iterator_t::next_batch(RawArray<board_value_t> batch) {
  for (const int i : range(batch.size()))
    batch[i] = next();
}

int shard_to_server_value(const board_t board, const int shard_value) {
  if (shard_value == 0) return 0;
  const bool black_to_move = count_stones(board) % 2 == 0;
  return (shard_value == 1) == black_to_move ? 1 : -1;
}

}  // namespace pentago
