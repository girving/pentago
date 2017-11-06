// Memory usage prediction
#pragma once

#include "pentago/end/load_balance.h"
#include "pentago/end/partition.h"
#include <cstdint>
namespace pentago {
namespace end {

struct block_partition_t;

// Estimate base memory usage of compute (ignoring active lines)
uint64_t base_compute_memory_usage(const int lines);

// Estimate block heap size required for a given partition on a given rank
uint64_t estimate_block_heap_size(const block_partition_t& partition, const int rank);

uint64_t max_rank_memory_usage(
    shared_ptr<const partition_t> prev_partition_, shared_ptr<const load_balance_t> prev_load_,
    const partition_t& partition, const load_balance_t& load);

}
}
