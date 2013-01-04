// Memory usage prediction

#include <pentago/mpi/partition.h>
#include <pentago/mpi/load_balance.h>
#include <pentago/mpi/block_store.h>
#include <pentago/mpi/flow.h>
#include <pentago/utility/memory.h>
#include <other/core/python/module.h>
#include <other/core/python/Ptr.h>
namespace pentago {
namespace mpi {

static uint64_t max_rank_memory_usage(Ptr<const partition_t> prev_partition_, Ptr<const load_balance_t> prev_load_, const partition_t& partition, const load_balance_t& load) {
  OTHER_ASSERT(!!prev_partition_==!!prev_load_);
  const auto prev_partition = prev_partition_?ref(prev_partition_):empty_partition(partition.ranks,0);
  const auto prev_load = prev_load_?ref(prev_load_):serial_load_balance(prev_partition);
  OTHER_ASSERT(prev_partition->ranks==partition.ranks);
  const auto partition_memory = memory_usage(prev_partition)+memory_usage(partition);
  const auto lines  = load.lines.max;
  const auto blocks = load.blocks.max+prev_load->blocks.max;
  const auto nodes  = load.block_nodes.max+prev_load->block_nodes.max;
  const auto memory = partition_memory // partition_t
                    + 2*sizeof(block_store_t) // block_store_t
                    + sizeof(block_info_t)*blocks // block_store_t.block_info
#if PENTAGO_MPI_COMPRESS
                    + sparse_store_t::estimate_peak_memory_usage(blocks,snappy_compression_estimate*sizeof(Vector<super_t,2>)*nodes) // block_store_t.store
#else
                    + sizeof(Vector<super_t,2>)*nodes // block_store_t.all_data
#endif
                    + sizeof(Vector<uint64_t,3>)*(prev_partition->sections->sections.size()+partition.sections->sections.size()) // block_store_t.section_counts
                    + sizeof(line_t)*lines+base_compute_memory_usage(lines); // line_t and line_data_t
  return memory;
}

}
}
using namespace pentago::mpi;

void wrap_predict() {
  OTHER_FUNCTION(max_rank_memory_usage)
}
