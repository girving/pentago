// Helper functions for testing purposes

#include <pentago/end/block_store.h>
namespace pentago {
namespace end {

GEODE_EXPORT Ref<block_store_t> meaningless_block_store(const partition_t& partition, const int rank, const int samples_per_section, compacting_store_t& store);

}
}
