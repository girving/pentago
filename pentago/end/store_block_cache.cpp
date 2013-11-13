// Cache of precomputed blocks from supertensor files

#include <pentago/end/store_block_cache.h>
#include <pentago/end/block_store.h>
#include <pentago/utility/debug.h>
#include <pentago/utility/large.h>
#include <pentago/utility/memory.h>
#include <geode/array/Array4d.h>
#include <geode/python/Class.h>
#include <geode/python/exceptions.h>
#include <geode/utility/Hasher.h>
#include <geode/utility/tr1.h>
namespace pentago {
namespace end {

using std::make_pair;

namespace {
struct store_block_cache_t : public block_cache_t {
  GEODE_DECLARE_TYPE(GEODE_NO_EXPORT)
  typedef block_cache_t Base;

  const Ref<const readable_block_store_t> blocks;
#if PENTAGO_MPI_COMPRESS
  const uint64_t memory_limit;
  mutable unordered_map<Tuple<section_t,Vector<uint8_t,4>>,Array<const Vector<super_t,2>,4>,Hasher> block_cache;
  mutable uint64_t free_memory;
#endif

protected:
  store_block_cache_t(const readable_block_store_t& blocks, const uint64_t memory_limit)
    : blocks(ref(blocks))
#if PENTAGO_MPI_COMPRESS
    , memory_limit(memory_limit)
    , free_memory(memory_limit)
#endif
  {
    GEODE_ASSERT(blocks.partition->ranks==1);
  }
public:

  int block_size() const {
    return end::block_size;
  }

  bool has_section(const section_t section) const {
    return blocks->sections->section_id.contains(section);
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
    const auto data = blocks->uncompress_and_get(section,block,unevent).copy();
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
GEODE_DEFINE_TYPE(store_block_cache_t)
}

Ref<const block_cache_t> store_block_cache(const readable_block_store_t& blocks, const uint64_t memory_limit) {
  return new_<store_block_cache_t>(blocks,memory_limit);
}

}
}
using namespace pentago::end;

void wrap_store_block_cache() {
  GEODE_FUNCTION(store_block_cache)
  Class<store_block_cache_t>("store_block_cache_t");
}
