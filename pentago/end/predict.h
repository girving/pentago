// Memory usage prediction
#pragma once

#include <geode/utility/config.h>
#include <stdint.h>
namespace pentago {
namespace end {

struct partition_t;

// Estimate base memory usage of compute (ignoring active lines)
GEODE_EXPORT uint64_t base_compute_memory_usage(const int lines);

// Estimate block heap size required for a given partition on a given rank
GEODE_EXPORT uint64_t estimate_block_heap_size(const partition_t& partition, const int rank);

}
}
