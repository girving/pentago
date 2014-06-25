// Asynchronous version of the block cache

#include <pentago/data/async_block_cache.h>
#include <pentago/high/index.h>
#include <pentago/utility/memory.h>
namespace pentago {

typedef async_block_cache_t::block_t block_t;

async_block_cache_t::async_block_cache_t(const uint64_t memory_limit)
  : memory_limit(memory_limit)
  , free_memory(memory_limit) {}

async_block_cache_t::~async_block_cache_t() {}

block_t async_block_cache_t::board_block(const high_board_t& board) {
  GEODE_ASSERT(!board.middle);
  // Account for global symmetries
  const auto flip_board = board.board; // Unlike block_cache_t::lookup, we don't need to flip the board
  const auto section = count(flip_board).standardize<8>();

  // More global symmetries
  const symmetry_t symmetry1(section.y,0);
  const auto board1 = transform_board(symmetry1,flip_board);

  // Account for local symmetries
  Vector<int,4> index;
  for  (int i=0;i<4;i++) {
    const int ir = rotation_minimal_quadrants_inverse[quadrant(board1,i)];
    index[i] = ir>>2;
  }

  // Load block if necessary
  const int block_size = pentago::end::block_size;
  const auto block = Vector<uint8_t,4>(index/block_size);
  return tuple(section.x,block);
}

bool async_block_cache_t::contains(const block_t block) const {
  return lru.get(block)!=0;
}

Unit async_block_cache_t::set(const block_t block, RawArray<const uint8_t> compressed) {
  const auto data = supertensor_index_t::unpack_block(block,compressed);
  free_memory -= memory_usage(data);
  while (free_memory < 0)
    free_memory += memory_usage(lru.drop().y);
  lru.add(block,data);
  return unit;
}

int async_block_cache_t::block_size() const {
  return pentago::end::block_size;
}

bool async_block_cache_t::has_section(const section_t section) const {
  return true; // Pretend we have all sections
}

// Copied from reader_block_cache_t
super_t async_block_cache_t::extract(const bool turn, const bool aggressive, const Vector<super_t,2>& data) const {
  return !turn?aggressive?data.x:~data.y  // Black to move
              :aggressive?data.y:~data.x; // White to move
}

RawArray<const Vector<super_t,2>,4> async_block_cache_t::load_block(const section_t section, const Vector<uint8_t,4> block) const {
  const auto p = lru.get(tuple(section,block));
  GEODE_ASSERT(p && p->flat.size(),format("async_block_cache_t::load_block: section %s, block %s %s",
                                          str(section),str(block),p?"pending":"missing"));
  return *p;
}

}
