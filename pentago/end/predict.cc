// Memory usage prediction

#include "pentago/end/predict.h"
#include "pentago/end/partition.h"
#include "pentago/end/line.h"
#include "pentago/end/load_balance.h"
#include "pentago/end/block_store.h"
#include "pentago/end/compute.h"
#include "pentago/utility/debug.h"
#include "pentago/utility/memory_usage.h"
namespace pentago {
namespace end {

uint64_t base_compute_memory_usage(const int lines) {
  return (2*sizeof(void*)+sizeof(line_data_t))*lines;
}

static uint64_t estimate_block_heap_size(const uint64_t blocks, const uint64_t nodes) {
  const uint64_t raw = sizeof(Vector<super_t,2>)*nodes;
#if PENTAGO_MPI_COMPRESS
  // In compressed mode, we tack an extra kilobyte onto each block for safety.
  return 1024*blocks+size_t(compacting_store_heap_ratio*snappy_compression_estimate*raw);
#else
  return raw;
#endif
}

uint64_t estimate_block_heap_size(const block_partition_t& partition, const int rank) {
  const auto counts = partition.rank_counts(rank);
  return estimate_block_heap_size(counts[0], counts[1]);
}

uint64_t max_rank_memory_usage(
    shared_ptr<const partition_t> prev_partition_, shared_ptr<const load_balance_t> prev_load_,
    const partition_t& partition, const load_balance_t& load) {
  GEODE_ASSERT(!!prev_partition_==!!prev_load_);
  const auto prev_partition = prev_partition_ ? prev_partition_ : empty_partition(partition.ranks, 0);
  const auto prev_load = prev_load_ ? prev_load_ : serial_load_balance(prev_partition);
  GEODE_ASSERT(prev_partition->ranks==partition.ranks);
  const auto partition_memory = memory_usage(prev_partition)+memory_usage(partition);
  const auto lines  = int(load.lines.max);
  const auto blocks = load.blocks.max+prev_load->blocks.max;
  const auto nodes  = load.block_nodes.max+prev_load->block_nodes.max;
  const auto heap   = estimate_block_heap_size(blocks,nodes);
  const auto memory = partition_memory // partition_t
                    + 2*sizeof(accumulating_block_store_t) // block_store
                    + sizeof(block_info_t)*blocks // block_store.block_info
                    + sizeof(Vector<uint64_t,3>)*(prev_partition->sections->sections.size()+partition.sections->sections.size()) // block_store.section_counts
                    + compacting_store_t::memory_usage(blocks,heap) // compacting_store_t
                    + sizeof(line_t)*lines+base_compute_memory_usage(lines); // line_t and line_data_t
  return memory;
}

}
}
