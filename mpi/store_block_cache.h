// Cache of precomputed blocks from an mpi block store
#pragma once

#include <pentago/block_cache.h>
namespace pentago {

namespace mpi {
class block_store_t;
}

// Generate a block cache from a block store
Ref<const block_cache_t> store_block_cache(const mpi::block_store_t& blocks, const uint64_t memory_limit);

}
