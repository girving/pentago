// Restart code for mpi computations

#include <pentago/end/restart_partition.h>
#include <pentago/end/blocks.h>
#include <pentago/utility/ceil_div.h>
#include <pentago/utility/index.h>
#include <pentago/utility/memory.h>
#include <geode/python/Class.h>
#include <geode/utility/Log.h>
namespace pentago {
namespace end {

GEODE_DEFINE_TYPE(restart_partition_t)
typedef restart_partition_t::Block Block;

static Ref<const sections_t> make_sections(RawArray<const Ref<const supertensor_reader_t>> tensors) {
  GEODE_ASSERT(tensors.size());
  const int slice = tensors[0]->header.stones;
  Array<section_t> sections;
  for (const auto& tensor : tensors)
    sections.append(tensor->header.section);
  return new_<sections_t>(slice,sections);
}

restart_partition_t::restart_partition_t(const int ranks, RawArray<const Ref<const supertensor_reader_t>> tensors)
  : restart_partition_t(ranks,make_sections(tensors),
                        partition_blocks(ranks,tensors,minimum_memory_limit(ranks,tensors))) {}

restart_partition_t::restart_partition_t(const int ranks, const sections_t& sections, Nested<const Block> partition)
  : Base(ranks,sections)
  , partition(partition) {
  // Build inverse partition
  GEODE_ASSERT(partition.size()==ranks);
  auto& inv = const_cast_(inv_partition);
  for (const int rank : range(ranks))
    for (const int i : range(partition.size(rank))) {
      const auto& I = partition(rank,i);
      inv.set(tuple(sections.sections[I.x],I.y),tuple(rank,local_id_t(i)));
    }
}

restart_partition_t::~restart_partition_t() {}

Nested<const Block> restart_partition_t::
partition_blocks(const int ranks, Tensors tensors, const uint64_t memory_limit) {
  thread_time_t time(partition_kind,unevent);
  Nested<Block,false> partition;
  partition.append_empty();
  int rank = 0;
  uint64_t used = 0;
  for (const int t : range(int(tensors.size())))
    for (const int i : range(tensors[t]->index.flat.size())) {
      const auto block = decompose(tensors[t]->index.shape,i);
      const uint64_t size = tensors[t]->index[block].uncompressed_size;
      while (used+size > memory_limit) {
        if (++rank == ranks) // If we run out of ranks, fail
          return Nested<Block>();
        used = 0;
        partition.append_empty();
      }
      used += size;
      partition.append_to_back(tuple(t,Vector<uint8_t,4>(block)));
    }
  return partition;
}

uint64_t restart_partition_t::minimum_memory_limit(const int ranks, Tensors tensors) {
  uint64_t count = 0, total = 0;
  for (const auto& tensor : tensors) {
    count += tensor->index.flat.size();
    for (const auto& blob : tensor->index.flat)
      total += blob.uncompressed_size;
  }
  GEODE_ASSERT(total);
  GEODE_ASSERT(count<numeric_limits<int>::max());
  uint64_t lo = ceil_div(total,ranks), hi = total;
  while (lo < hi) {
    const auto mid = (lo+hi)/2;
    if (partition_blocks(ranks,tensors,mid).size())
      hi = mid;
    else
      lo = mid+1;
  }
  GEODE_ASSERT(lo<numeric_limits<int>::max());
  return lo;
}

uint64_t restart_partition_t::memory_usage() const {
  return sizeof(restart_partition_t)
       + pentago::memory_usage(partition)
       + pentago::memory_usage(inv_partition);
}

Array<const local_block_t> restart_partition_t::rank_blocks(const int rank) const {
  GEODE_ASSERT(partition.valid(rank));
  Array<local_block_t> blocks(partition.size(rank));
  for (const int i : range(partition.size(rank))) {
    const auto& I = partition(rank,i);
    auto& b = blocks[i];
    b.local_id = local_id_t(i);
    b.section = sections->sections[I.x];
    b.block = I.y;
  }
  return blocks;
}

Vector<uint64_t,2> restart_partition_t::rank_counts(const int rank) const {
  uint64_t nodes = 0;
  const auto blocks = rank_blocks(rank);
  for (const auto& block : blocks)
    nodes += block_shape(block.section.shape(),block.block).product();
  return vec(uint64_t(blocks.size()),nodes);
}

Tuple<int,local_id_t> restart_partition_t::find_block(const section_t section, const Vector<uint8_t,4> block) const {
  return inv_partition.get(tuple(section,block));
}

Tuple<section_t,Vector<uint8_t,4>> restart_partition_t::rank_block(const int rank, const local_id_t local_id) const {
  GEODE_ASSERT(partition.valid(rank,local_id.id));
  const auto& I = partition(rank,local_id.id);
  return tuple(sections->sections[I.x],I.y);
}

}
}
