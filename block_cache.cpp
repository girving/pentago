// Cache of precomputed blocks from supertensor files

#include <pentago/block_cache.h>
#include <pentago/section.h>
#include <pentago/supertensor.h>
#include <pentago/symmetry.h>
#include <pentago/mpi/block_store.h>
#include <pentago/utility/debug.h>
#include <pentago/utility/memory.h>
#include <pentago/utility/large.h>
#include <other/core/python/Class.h>
#include <other/core/python/stl.h>
#include <other/core/utility/Hasher.h>
#include <other/core/utility/Log.h>
#include <tr1/unordered_map>
namespace pentago {

OTHER_DEFINE_TYPE(block_cache_t)
using std::tr1::unordered_map;
using std::make_pair;
using Log::cout;
using std::endl;

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
  if (!has_section(section.x))
    return false;

  // More global symmetries
  const symmetry_t symmetry1(section.y,0);
  const auto board1 = transform_board(symmetry1,flip_board);

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
  const auto block = Vector<uint8_t,4>(index/block_size);
  const auto block_data = load_block(section.x,block);
  const auto I = index-block_size*Vector<int,4>(block);
  OTHER_ASSERT(block_data.valid(I));
  const auto& data = block_data[I];

  // Extract the part we want
  wins = transform_super(symmetry,extract(turn,aggressive,data));
  return true;
}

namespace {
struct reader_block_cache_t : public block_cache_t {
  OTHER_DECLARE_TYPE(OTHER_NO_EXPORT)
  typedef block_cache_t Base;

  const uint64_t memory_limit;
  unordered_map<section_t,Ref<const supertensor_reader_t>,Hasher> readers;
  mutable unordered_map<Tuple<section_t,Vector<uint8_t,4>>,Array<const Vector<super_t,2>,4>,Hasher> block_cache;
  mutable uint64_t free_memory;

protected:
  reader_block_cache_t(const vector<Ref<const supertensor_reader_t>> reader_list, const uint64_t memory_limit)
    : memory_limit(memory_limit)
    , free_memory(memory_limit) {
    for (const auto& reader : reader_list) {
      OTHER_ASSERT((int)reader->header.block_size==block_size);
      readers.insert(make_pair(reader->header.section,reader));
    }
  }
public:

  bool has_section(const section_t section) const {
    return readers.count(section);
  }

  super_t extract(const bool turn, const bool aggressive, const Vector<super_t,2>& data) const {
    return turn?aggressive?data.x:~data.y  // Black to move
               :aggressive?data.y:~data.x; // White to move
  }

  RawArray<const Vector<super_t,2>,4> load_block(const section_t section, const Vector<uint8_t,4> block) const {
    const auto& reader = *readers.find(section)->second;
    const auto key = tuple(reader.header.section,block);
    const auto it = block_cache.find(key);
    if (it != block_cache.end())
      return it->second;
    const auto data = reader.read_block(block);
    const auto memory = memory_usage(data);
    if (free_memory < memory)
      THROW(RuntimeError,"reader_block_cache_t: memory limit of %s exceeded (%zu blocks loaded)",large(memory_limit),block_cache.size());
    free_memory -= memory;
    block_cache.insert(make_pair(key,data));
    return data;
  }
};
OTHER_DEFINE_TYPE(reader_block_cache_t)

struct store_block_cache_t : public block_cache_t {
  OTHER_DECLARE_TYPE(OTHER_NO_EXPORT)
  typedef block_cache_t Base;

  const Ref<const mpi::block_store_t> blocks;
#if PENTAGO_MPI_COMPRESS
  const uint64_t memory_limit;
  mutable unordered_map<Tuple<section_t,Vector<uint8_t,4>>,Array<const Vector<super_t,2>,4>,Hasher> block_cache;
  mutable uint64_t free_memory;
#endif

protected:
  store_block_cache_t(const mpi::block_store_t& blocks, const uint64_t memory_limit)
    : blocks(ref(blocks))
#if PENTAGO_MPI_COMPRESS
    , memory_limit(memory_limit)
    , free_memory(memory_limit)
#endif
  {
    OTHER_ASSERT(blocks.partition->ranks==1);
  }
public:

  bool has_section(const section_t section) const {
    return blocks->partition->section_id.contains(section);
  }

  super_t extract(const bool turn, const bool aggressive, const Vector<super_t,2>& data) const {
    return aggressive?data.x:data.y;
  }

  RawArray<const Vector<super_t,2>,4> load_block(const section_t section, const Vector<uint8_t,4> block) const {
#if PENTAGO_MPI_COMPRESS
    const auto key = tuple(section,block);
    const auto it = block_cache.find(key);
    if (it != block_cache.end())
      return it->second;
    const auto data = blocks->uncompress_and_get(section,block,unevent);
    const auto memory = memory_usage(data);
    if (free_memory < memory)
      THROW(RuntimeError,"store_block_cache_t: memory limit of %s exceeded (%zu blocks loaded)",large(memory_limit),block_cache.size());
    free_memory -= memory;
    block_cache.insert(make_pair(key,data));
    return data;
#else
    return blocks->get_raw(section,block);
#endif
  }
};
OTHER_DEFINE_TYPE(store_block_cache_t)
}

Ref<const block_cache_t> reader_block_cache(const vector<Ref<const supertensor_reader_t>> readers, const uint64_t memory_limit) {
  return new_<reader_block_cache_t>(readers,memory_limit);
}

Ref<const block_cache_t> store_block_cache(const mpi::block_store_t& blocks, const uint64_t memory_limit) {
  return new_<store_block_cache_t>(blocks,memory_limit);
}

}
using namespace pentago;

void wrap_block_cache() {
  OTHER_FUNCTION(reader_block_cache)
  OTHER_FUNCTION(store_block_cache)
  Class<block_cache_t>("block_cache_t");
  Class<reader_block_cache_t>("reader_block_cache_t");
  Class<store_block_cache_t>("store_block_cache_t");
}
