// Cache of precomputed blocks from supertensor files

#include "pentago/data/block_cache.h"
#include "pentago/data/lru.h"
#include "pentago/data/supertensor.h"
#include "pentago/base/section.h"
#include "pentago/base/symmetry.h"
#include "pentago/utility/debug.h"
#include "pentago/utility/memory_usage.h"
#include "pentago/utility/large.h"
#include "pentago/utility/const_cast.h"
namespace pentago {

using std::make_pair;
using std::make_shared;
using std::endl;
using std::get;

block_cache_t::block_cache_t() {}
block_cache_t::~block_cache_t() {}

bool block_cache_t::lookup(const bool aggressive, const side_t side0, const side_t side1, super_t& wins) const {
  return lookup(aggressive,pack(side0,side1),wins);
}

bool block_cache_t::lookup(const bool aggressive, const board_t board, super_t& wins) const {
  // Account for global symmetries
  const bool turn = count(board).sum()&1;
  const auto flip_board = pentago::flip_board(board,turn);
  const auto section = count(flip_board).standardize<8>();

  // Do we have access to this section?
  if (!has_section(get<0>(section)))
    return false;

  // More global symmetries
  const symmetry_t symmetry1(get<1>(section), 0);
  const auto board1 = transform_board(symmetry1, flip_board);

  // Account for local symmetries
  Vector<int,4> index;
  uint8_t local_rotations = 0;
  for  (int i=0;i<4;i++) {
    const int ir = rotation_minimal_quadrants_inverse[quadrant(board1,i)];
    index[i] = ir>>2;
    local_rotations |= (ir&3)<<2*i;
  }
  const auto symmetry = symmetry1.inverse()*local_symmetry_t(local_rotations);

  // Load block if necessary
  const int block_size = this->block_size();
  const auto block = Vector<uint8_t,4>(index/block_size);
  const auto block_data = load_block(get<0>(section), block);
  const auto I = index-block_size*Vector<int,4>(block);
  GEODE_ASSERT(block_data.valid(I));
  const auto& data = block_data[I];

  // Extract the part we want
  wins = transform_super(symmetry,extract(turn,aggressive,data));
  return true;
}

namespace {
struct empty_block_cache_t : public block_cache_t {
  typedef block_cache_t Base;
public:
  int block_size() const {
    return 8;
  }

  bool has_section(const section_t section) const {
    return false;
  }

  super_t extract(const bool turn, const bool aggressive, const Vector<super_t,2>& data) const {
    THROW(AssertionError, "Trying to access into an empty_block_cache_t");
  }

  RawArray<const Vector<super_t,2>,4> load_block(const section_t section, const Vector<uint8_t,4> block) const {
    THROW(AssertionError, "Trying to access into an empty_block_cache_t");
  }
};

struct reader_block_cache_t : public block_cache_t {
  typedef block_cache_t Base;

  const uint64_t memory_limit;
  const int block_size_;
  const unordered_map<section_t,shared_ptr<const supertensor_reader_t>> readers;
  mutable lru_t<tuple<section_t,Vector<uint8_t,4>>,Array<const Vector<super_t,2>,4>> lru;
  mutable int64_t free_memory; // signed so it can go temporarily below zero

public:
  reader_block_cache_t(const vector<shared_ptr<const supertensor_reader_t>> reader_list,
                       const uint64_t memory_limit)
    : memory_limit(memory_limit)
    , block_size_(0) // filled in below if reader_list is nonempty, never used if reader_list is empty
    , free_memory(memory_limit) {
    for (const auto& reader : reader_list) {
      if (!block_size_)
        const_cast<int&>(block_size_) = reader->header.block_size;
      GEODE_ASSERT((int)reader->header.block_size==block_size_);
      const_cast_(readers).insert(make_pair(reader->header.section,reader));
    }
  }

  int block_size() const {
    return block_size_;
  }

  bool has_section(const section_t section) const {
    return readers.count(section);
  }

  super_t extract(const bool turn, const bool aggressive, const Vector<super_t,2>& data) const {
    return !turn ? aggressive ? data[0] : ~data[1]  // Black to move
                 : aggressive ? data[1] : ~data[0]; // White to move
  }

  RawArray<const Vector<super_t,2>,4> load_block(const section_t section, const Vector<uint8_t,4> block) const {
    const auto& reader = *readers.find(section)->second;
    const auto key = make_tuple(reader.header.section, block);
    if (const auto p = lru.get(key))
      return *p;
    const auto data = reader.read_block(block);
    free_memory -= memory_usage(data);
    while (free_memory < 0)
      free_memory += memory_usage(get<1>(lru.drop()));
    lru.add(key,data);
    return data;
  }
};
}

shared_ptr<const block_cache_t> empty_block_cache() {
  return make_shared<empty_block_cache_t>();
}

shared_ptr<const block_cache_t>
reader_block_cache(const vector<shared_ptr<const supertensor_reader_t>> readers,
                   const uint64_t memory_limit) {
  return make_shared<reader_block_cache_t>(readers, memory_limit);
}

}
