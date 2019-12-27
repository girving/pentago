// Cache of precomputed blocks from supertensor files
//
// A block_cache_t serves as an intermediary between the forward search engines
// and endgame information.  It was originally written for unit test purposes,
// allowing the backward engine to be validated against forward search, but now
// is also used by the web backend to manage cached portions of the data set.
#pragma once

#include "pentago/base/board.h"
#include "pentago/base/superscore.h"
#include <boost/core/noncopyable.hpp>
#include <vector>
namespace pentago {

struct section_t;
struct supertensor_reader_t;
using std::vector;

struct block_cache_t : public boost::noncopyable {
public:
  block_cache_t();
  virtual ~block_cache_t();

  // Warning: Very slow, use only inside low depth searches.  board has player 0 to move.
  bool lookup(const bool aggressive, const board_t board, super_t& wins) const;
  bool lookup(const bool aggressive, const side_t side0, const side_t side1, super_t& wins) const;

private:
  virtual int block_size() const = 0;
  virtual bool has_section(const section_t section) const = 0; 
  virtual super_t extract(const bool turn, const bool aggressive, const Vector<super_t,2>& data) const = 0;
  virtual RawArray<const Vector<super_t,2>,4> load_block(const section_t section, const Vector<uint8_t,4> block) const = 0;
};

// An empty block cache
shared_ptr<const block_cache_t> empty_block_cache();

// Generate a block cache from one or more supertensor files
shared_ptr<const block_cache_t> reader_block_cache(
    const vector<shared_ptr<const supertensor_reader_t>> readers, const uint64_t memory_limit);

}
