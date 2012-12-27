// Memory usage prediction

#include <pentago/mpi/partition.h>
#include <pentago/mpi/block_store.h>
#include <pentago/mpi/flow.h>
#include <pentago/utility/memory.h>
#include <other/core/python/module.h>
#include <other/core/python/Ptr.h>
namespace pentago {
namespace mpi {

static uint64_t max_rank_memory_usage(Ptr<const partition_t> prev_partition_, const partition_t& partition) {
  const auto prev_partition = prev_partition_?prev_partition_:new_<partition_t>(partition.ranks,0,Array<section_t>());
  OTHER_ASSERT(prev_partition->ranks==partition.ranks);
  const auto partition_memory = memory_usage(prev_partition)+memory_usage(partition);
  auto prev_start = prev_partition->rank_offsets(0),
       start = partition.rank_offsets(0);
  uint64_t max_memory = 0;
  for (int rank=0;rank<partition.ranks;rank++) {
    const auto prev_end = prev_partition->rank_offsets(rank+1),
               end = partition.rank_offsets(rank+1);
    const auto lines = partition.rank_count_lines(rank,true)+partition.rank_count_lines(rank,false);
    const auto counts = prev_end-prev_start+end-start;
    const auto memory = partition_memory // partition_t
                      + 2*sizeof(block_store_t) // block_store_t
                      + sizeof(block_info_t)*counts.x // block_store_t.block_info
#if PENTAGO_MPI_COMPRESS
                      + sparse_store_t::estimate_peak_memory_usage(counts.x,snappy_compression_estimate*sizeof(Vector<super_t,2>)*counts.y) // block_store_t.store
#else
                      + sizeof(Vector<super_t,2>)*counts.y // block_store_t.all_data
#endif
                      + sizeof(Vector<uint64_t,3>)*(prev_partition->sections.size()+partition.sections.size()) // block_store_t.section_counts
                      + sizeof(line_t)*lines+base_compute_memory_usage(lines); // line_t and line_data_t
    max_memory = max(max_memory,memory);
    prev_start = prev_end;
    start = end;
  }
  return max_memory;
}

}
}
using namespace pentago::mpi;

void wrap_predict() {
  OTHER_FUNCTION(max_rank_memory_usage)
}
