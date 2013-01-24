// Cache of precomputed blocks from supertensor files
#pragma once

#include <pentago/board.h>
#include <pentago/superscore.h>
#include <other/core/python/Object.h>
#include <vector>
namespace pentago {

struct section_t;
struct supertensor_reader_t;
using std::vector;

struct OTHER_EXPORT block_cache_t : public Object {
  OTHER_DECLARE_TYPE(OTHER_EXPORT)

protected:
  OTHER_EXPORT block_cache_t();
public:
  OTHER_EXPORT ~block_cache_t();

  // Warning: Very slow, use only inside low depth searches
  OTHER_EXPORT bool lookup(const bool aggressive, const board_t board, super_t& wins) const;
  OTHER_EXPORT bool lookup(const bool aggressive, const side_t side0, const side_t side1, super_t& wins) const;

private:
  virtual int block_size() const = 0;
  virtual bool has_section(const section_t section) const = 0; 
  virtual super_t extract(const bool turn, const bool aggressive, const Vector<super_t,2>& data) const = 0;
  virtual RawArray<const Vector<super_t,2>,4> load_block(const section_t section, const Vector<uint8_t,4> block) const = 0;
};

// Generate a block cache from one or more supertensor files
OTHER_EXPORT Ref<const block_cache_t> reader_block_cache(const vector<Ref<const supertensor_reader_t>> readers, const uint64_t memory_limit);

}
