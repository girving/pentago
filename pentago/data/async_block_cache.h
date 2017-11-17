// Asynchronous version of the block cache
//
// A version of block_cache_t that does not know how to get its own data.
// This is used to lift control flow into the asynchronous world of node.js,
// which is responsible to actually retrieving the data via asynchronous
// range requests.
#pragma once

#include "pentago/data/block_cache.h"
#include "pentago/data/lru.h"
#include "pentago/base/section.h"
#include "pentago/base/symmetry.h"
#include "pentago/end/config.h"
#include "pentago/high/board.h"
#include "pentago/utility/array.h"
#include "pentago/utility/unit.h"
namespace pentago {

struct async_block_cache_t : public block_cache_t {
  typedef block_cache_t Base;
  typedef tuple<section_t,Vector<uint8_t,4>> block_t;

private:
  lru_t<block_t,Array<const Vector<super_t,2>,4>> lru; // Maybe be an empty array, indicating pending
  int64_t free_memory; // signed so it can go temporarily below zero

public:
  async_block_cache_t(const uint64_t memory_limit);
  ~async_block_cache_t();

  // Look up the section and block containing a given board.
  // Code copied from block_cache_t::lookup due to laziness.
  // The only difference is that the board isn't flipped.
  static block_t board_block(const high_board_t& board);

  bool contains(const block_t block) const;
  unit_t set(const block_t block, RawArray<const uint8_t> compressed);

  // Make set flaky so that unit tests can test error paths
  static unit_t set_flaky(const double flake_probability);

private:
  int block_size() const;
  bool has_section(const section_t section) const;
  super_t extract(const bool turn, const bool aggressive, const Vector<super_t,2>& data) const;
  RawArray<const Vector<super_t,2>,4> load_block(const section_t section, const Vector<uint8_t,4> block) const;
};

}
