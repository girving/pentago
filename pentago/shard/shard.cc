// Pseudorandom shard mapping for Pentago positions

#include "pentago/shard/shard.h"
#include "pentago/shard/shard_permute.h"
#include "pentago/base/all_boards.h"
#include "pentago/base/blocks.h"
#include "pentago/data/file.h"
#include "pentago/utility/char_view.h"
#include "pentago/utility/const_cast.h"
#include "pentago/utility/debug.h"
#include "pentago/utility/endian.h"
#include "pentago/utility/index.h"
#include "pentago/utility/permute.h"
#include "pentago/utility/range.h"
#include <algorithm>
#include <cstring>
#include <sys/stat.h>
namespace pentago {

using std::get;
using std::make_unique;
using std::memcpy;
using std::upper_bound;

static inline uint64_t mulhi_u64(const uint64_t a, const uint64_t b) {
  return uint64_t((__uint128_t(a) * b) >> 64);
}

static inline uint64_t fast_div(const uint64_t x, const recip_t& r) {
  if (!r.magic) return x >> r.shift;
  if (!r.overflow) return mulhi_u64(x, r.magic) >> r.shift;
  const uint64_t t = mulhi_u64(x, r.magic);
  return (t + ((x - t) >> 1)) >> (r.shift - 1);
}

static recip_t compute_recip(const int d) {
  if (d <= 1) return {0, 0, false};
  if ((d & (d - 1)) == 0) return {0, __builtin_ctz(d), false};
  const int l = 64 - __builtin_clzll(uint64_t(d) - 1);  // ceil(log2(d))
  const __uint128_t m = (__uint128_t(1) << (64 + l)) / d + 1;
  if (m <= ~uint64_t(0)) return {uint64_t(m), l, false};
  return {uint64_t(m), l, true};  // overflow: add-one variant
}

shard_mapping_t::shard_mapping_t(const int slice)
    : slice(slice), permute(slice) {
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

  // Precompute per-section info for fast board() lookup
  Array<section_info_t> info(sections.size(), uninit);
  for (const int i : range(sections.size())) {
    auto& si = info[i];
    for (const int q : range(4)) {
      si.rmin[q] = get<0>(rotation_minimal_quadrants(sections[i].counts[q]));
      si.shape[q] = si.rmin[q].size();
    }
    si.recip[0] = compute_recip(si.shape[3]);
    si.recip[1] = compute_recip(si.shape[2]);
    si.recip[2] = compute_recip(si.shape[1]);
  }
  const_cast_(section_info) = info;

  // Build direct lookup table: section_lookup[linear >> shift] gives the section
  // containing the start of that bucket.  For any linear value in the bucket,
  // the true section is this value or a small number of steps forward.
  const uint64_t t = total();
  if (t <= 1) {
    section_shift = 0;
    const_cast_(section_lookup) = Array<uint16_t>(1);
  } else {
    section_shift = std::max(0, int(64 - __builtin_clzll(t - 1)) - 16);
    const int table_size = int((t - 1) >> section_shift) + 1;
    Array<uint16_t> lookup(table_size, uninit);
    int si = 0;
    for (const int i : range(table_size)) {
      const uint64_t bucket_start = uint64_t(i) << section_shift;
      while (si + 1 < sections.size() && offsets[si + 1] <= bucket_start)
        si++;
      lookup[i] = uint16_t(si);
    }
    const_cast_(section_lookup) = lookup;
  }
}

shard_mapping_t::~shard_mapping_t() {}

int shard_mapping_t::section_index(const section_t section) const {
  const auto it = section_id.find(section);
  GEODE_ASSERT(it != section_id.end());
  return it->second;
}

uint64_t shard_mapping_t::forward(const section_t section, const Vector<int,4> index,
                                  const local_symmetry_t rotation) const {
  const int si = section_index(section);
  const auto shape = section.shape();
  GEODE_ASSERT(valid(shape, index));
  const uint64_t pos = index64(shape, index);
  const uint64_t linear = offsets[si] + pos * 256 + rotation.local;
  return permute.forward(linear);
}

int shard_mapping_t::find_section(const uint64_t linear) const {
  int si = section_lookup[linear >> section_shift];
  while (offsets[si + 1] <= linear) si++;
  return si;
}

shard_mapping_t::location_t shard_mapping_t::inverse(const uint64_t shuffled) const {
  GEODE_ASSERT(shuffled < total());
  const uint64_t linear = permute.inverse(shuffled);

  const int si = find_section(linear);
  GEODE_ASSERT(unsigned(si) < unsigned(sections.size()));

  // Decompose within section
  const uint64_t within = linear - offsets[si];
  const uint64_t pos = within / 256;
  const int rot = int(within % 256);
  const auto shape = sections[si].shape();
  return {sections[si], decompose64(shape, pos), local_symmetry_t(uint8_t(rot))};
}

board_t shard_mapping_t::board(const uint64_t shuffled) const {
  GEODE_ASSERT(shuffled < total());
  const uint64_t linear = permute.inverse(shuffled);

  const int si = find_section(linear);
  GEODE_ASSERT(unsigned(si) < unsigned(sections.size()));

  // Decompose within section
  const uint64_t within = linear - offsets[si];
  const uint64_t pos = within >> 8;
  const int rot = int(within & 255);

  // Fast decompose using precomputed reciprocals (replaces decompose64 + shape())
  const auto& info = section_info[si];
  int index[4];
  uint64_t rem = pos;
  for (int i = 0; i < 3; i++) {
    const int d = info.shape[3 - i];
    const auto& r = info.recip[i];
    const uint64_t q = d <= 1 ? rem : fast_div(rem, r);
    index[3 - i] = d <= 1 ? 0 : int(rem - q * d);
    rem = q;
  }
  index[0] = int(rem);

  // Direct quadrant lookup from precomputed rmin arrays
  const auto b = quadrants(
      info.rmin[0][index[0]], info.rmin[1][index[1]],
      info.rmin[2][index[2]], info.rmin[3][index[3]]);
  return transform_board(local_symmetry_t(uint8_t(rot)), b);
}

shard_header_t::shard_header_t()
    : version(2), max_slice(0), shard_id(0), total_shards(0) {
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
  GEODE_ASSERT(h.version == 2);
  GEODE_ASSERT(h.max_slice <= 18);
  GEODE_ASSERT(h.shard_id < h.total_shards);
  return h;
}

Array<uint8_t> serialize_shard(const shard_header_t& header,
                               RawArray<const arithmetic_t> groups) {
  const int n = header.max_slice + 1;
  GEODE_ASSERT(groups.size() == n);

  // Compute offsets from serialized sizes
  uint64_t total = shard_header_t::header_size + (n + 1) * 4;
  Array<uint32_t> offsets(n + 1, uninit);
  for (const int i : range(n)) {
    GEODE_ASSERT(total <= numeric_limits<uint32_t>::max());
    offsets[i] = uint32_t(total);
    total += groups[i].serialized_size();
  }
  GEODE_ASSERT(total <= numeric_limits<uint32_t>::max());
  offsets[n] = uint32_t(total);

  // Assemble into a single buffer
  Array<uint8_t> buf(int(total), uninit);
  header.pack(buf.slice(0, shard_header_t::header_size));
  to_little_endian_inplace(offsets);
  memcpy(buf.data() + shard_header_t::header_size, offsets.data(), (n + 1) * 4);
  for (const int i : range(n)) {
    const auto group_buf = arithmetic_serialize(groups[i]);
    memcpy(buf.data() + little_to_native_endian(offsets[i]), group_buf.data(), group_buf.size());
  }
  return buf;
}

void write_shard(const string& path, const shard_header_t& header,
                 RawArray<const arithmetic_t> groups) {
  const auto buf = serialize_shard(header, groups);
  const auto fd = write_local_file(path);
  const auto err = fd->pwrite(buf, 0);
  GEODE_ASSERT(err.empty(), err);
}

shard_file_t::shard_file_t(Array<const uint8_t> data)
    : data(data) {
  GEODE_ASSERT(data.size() >= shard_header_t::header_size);
  header = shard_header_t::unpack(data.slice(0, shard_header_t::header_size));

  // Read offset table
  const int n = header.max_slice + 1;
  const int table_size = (n + 1) * 4;
  GEODE_ASSERT(data.size() >= shard_header_t::header_size + table_size);
  Array<uint32_t> offs(n + 1, uninit);
  memcpy(offs.data(), data.data() + shard_header_t::header_size, table_size);
  to_little_endian_inplace(offs);
  group_offsets = offs;
}

shard_file_t::shard_file_t(const string& path) : shard_file_t([&]() {
  struct stat st;
  GEODE_ASSERT(stat(path.c_str(), &st) == 0, path);
  Array<uint8_t> buf(int(st.st_size), uninit);
  const auto fd = read_local_file(path);
  const auto err = fd->pread(buf, 0);
  GEODE_ASSERT(err.empty(), err);
  return Array<const uint8_t>(buf);
}()) {}

shard_file_t::~shard_file_t() {}

arithmetic_t shard_file_t::read_group(const int slice) const {
  GEODE_ASSERT(0 <= slice && slice <= int(header.max_slice));
  const uint32_t start = group_offsets[slice];
  const uint32_t end = group_offsets[slice + 1];
  GEODE_ASSERT(start < end && end <= uint32_t(data.size()));
  return arithmetic_deserialize(data.slice(start, end));
}

string shard_filename(const int shards, const int shard) {
  const int w = int(tfm::format("%d", shards - 1).size());
  return tfm::format("shard-%0*d-of-%0*d.pentago.shard", w, shard, w, shards - 1);
}

int parse_shard_total(const string& path) {
  // Extract total_shards from "shard-NNNNNNN-of-NNNNNNN.pentago.shard"
  const auto base = path.substr(path.rfind('/') + 1);
  const auto of_pos = base.find("-of-");
  GEODE_ASSERT(of_pos != string::npos);
  const auto dot_pos = base.find('.', of_pos);
  return std::stoi(base.substr(of_pos + 4, dot_pos - of_pos - 4)) + 1;
}

static vector<string> build_shard_paths(const string& dir, const int total_shards,
                                        const Range<int> shard_range) {
  vector<string> paths;
  paths.reserve(shard_range.size());
  for (const int id : shard_range)
    paths.push_back(tfm::format("%s/%s", dir, shard_filename(total_shards, id)));
  return paths;
}

shard_iterator_t::shard_iterator_t(vector<string> shard_paths, const uint128_t seed,
                                   const int max_epochs)
    : shard_paths(std::move(shard_paths)),
      total_shards(parse_shard_total(this->shard_paths.at(0))),
      locator(total_shards, range(0, 1)),  // dummy range; only shard_size/shuffled_index used
      max_epochs(max_epochs), random(seed),
      shard_cursor(0), epoch_key(random.bits<uint128_t>()),
      epoch_count_(0), done_(false), total_remaining(0) {
  GEODE_ASSERT(!this->shard_paths.empty());
  load_next_shard();
}

shard_iterator_t::shard_iterator_t(const string& dir, const int total_shards,
                                   const Range<int> shard_range, const uint128_t seed,
                                   const int max_epochs)
    : shard_iterator_t(build_shard_paths(dir, total_shards, shard_range), seed, max_epochs) {}

shard_iterator_t::~shard_iterator_t() {}

void shard_iterator_t::load_next_shard() {
  const int n = int(shard_paths.size());
  if (shard_cursor >= n) {
    epoch_count_++;
    if (max_epochs >= 0 && epoch_count_ >= max_epochs) {
      done_ = true;
      return;
    }
    epoch_key = random.bits<uint128_t>();
    shard_cursor = 0;
  }
  const int permuted = int(random_permute(uint64_t(n), epoch_key,
                                          uint64_t(shard_cursor++)));
  const shard_file_t sf(shard_paths[permuted]);

  // Initialize slices lazily on first shard load
  if (slices.empty()) {
    const int ns = sf.header.max_slice + 1;
    slices.reserve(ns);
    for (const int s : range(ns))
      slices.emplace_back(s);
  }
  GEODE_ASSERT(sf.header.max_slice + 1 == uint32_t(slices.size()) &&
               sf.header.total_shards == uint32_t(total_shards));

  // Decode all groups and set up per-slice state
  const int shard_id = int(sf.header.shard_id);
  total_remaining = 0;
  for (auto& slice : slices) {
    slice.decoded.emplace(arithmetic_decode(sf.read_group(slice.mapping.slice)));
    slice.remaining = locator.shard_size(slice.mapping.total(), shard_id);
    slice.cursor = 0;
    slice.shard_range_lo = shard_id;
    total_remaining += slice.remaining;
  }
}

board_value_t shard_iterator_t::next() {
  GEODE_ASSERT(!done_);

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
  const board_t board = slice.mapping.board(
      locator.shuffled_index(int(slice.shard_range_lo), slice.cursor));

  // Advance
  slice.cursor++;
  slice.remaining--;
  total_remaining--;

  // Eagerly load next shard so done() is immediately accurate
  if (total_remaining == 0)
    load_next_shard();

  return {board, value};
}

int shard_iterator_t::next_batch(RawArray<board_value_t> batch) {
  for (const int i : range(batch.size())) {
    if (done_) return i;
    batch[i] = next();
  }
  return batch.size();
}

int shard_to_server_value(const board_t board, const int shard_value) {
  if (shard_value == 0) return 0;
  const bool black_to_move = count_stones(board) % 2 == 0;
  return (shard_value == 1) == black_to_move ? 1 : -1;
}

void scatter_block(
    const shard_mapping_t& mapping,
    const int total_shards,
    const Range<int> shard_range,
    RawArray<ternaries_t> buffers,
    const section_t section,
    const Vector<uint8_t,4> block,
    RawArray<const Vector<super_t,2>,4> data) {
  // Hoist per-block invariants out of position and rotation loops
  const int si = mapping.section_index(section);
  const auto shape = section.shape();
  const uint64_t offset = mapping.offsets[si];
  const auto base_index = Vector<int,4>(block) * block_size;
  const auto block_shape = data.shape();
  const auto perm = mapping.permute;
  const shard_locator_t locator(total_shards, shard_range);
  for (const int i0 : range(block_shape[0]))
    for (const int i1 : range(block_shape[1]))
      for (const int i2 : range(block_shape[2]))
        for (const int i3 : range(block_shape[3])) {
          const auto index = base_index + vec(i0, i1, i2, i3);
          const auto& entry = data(i0, i1, i2, i3);
          const auto& black_wins = entry[0];
          const auto& white_wins = entry[1];
          // Hoist per-position index computation out of rotation loop
          const uint64_t pos = index64(shape, index);
          const uint64_t base_linear = offset + pos * 256;
#if PENTAGO_SSE
          const __m256i off0 = _mm256_setr_epi64x(0, 1, 2, 3);
          const __m256i off1 = _mm256_setr_epi64x(4, 5, 6, 7);
          for (int r = 0; r < 256; r += 8) {
            const __m256i base = _mm256_set1_epi64x(base_linear + r);
            const auto shuffled = perm.forward8({_mm256_add_epi64(base, off0),
                                                  _mm256_add_epi64(base, off1)});
            auto mask = locator.in_range8(shuffled);
            if (!mask) continue;
            alignas(32) uint64_t sv[8];
            _mm256_store_si256((__m256i*)&sv[0], shuffled.v0);
            _mm256_store_si256((__m256i*)&sv[4], shuffled.v1);
            for (; mask; mask.advance()) {
              const int j = mask.index();
              const int s = locator.shard(sv[j]);
              const uint64_t p = locator.position(sv[j]);
              buffers[s - shard_range.lo].atomic_set_from_zero(
                  p, black_wins(r + j) + 2 * white_wins(r + j));
            }
          }
#else
          for (const int r : range(256)) {
            const uint64_t shuffled = perm.forward(base_linear + r);
            const int s = locator.shard(shuffled);
            if (s < shard_range.lo || s >= shard_range.hi)
              continue;
            const uint64_t p = locator.position(shuffled);
            buffers[s - shard_range.lo].atomic_set_from_zero(
                p, black_wins(r) + 2 * white_wins(r));
          }
#endif
        }
}

}  // namespace pentago
