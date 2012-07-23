// Data structure keeping track of all blocks we own

#include <pentago/mpi/block_store.h>
#include <pentago/mpi/utility.h>
#include <pentago/utility/aligned.h>
#include <pentago/utility/memory.h>
#include <other/core/python/Class.h>
#include <other/core/utility/const_cast.h>
namespace pentago {
namespace mpi {

OTHER_DEFINE_TYPE(block_store_t)

block_store_t::block_store_t(const partition_t& partition, const int rank, Array<const line_t> lines)
  : partition(ref(partition))
  , rank(rank)
  , first(partition.rank_offsets(rank))
  , last(partition.rank_offsets(rank+1))
  , required_contributions(0) // set below
  , missing_contributions(0) // set below
  , complete(false) {

  // Make sure 32-bit array indices suffice
  OTHER_ASSERT(last.y-first.y<(1<<31));

  // Compute map from local block id to local offset
  int block_id = 0, node = 0;
  Array<block_info_t> block_info(last.x-first.x+1,false);
  block_info[0].offset = 0;
  for (auto line : lines)
    for (int i : range(line.length)) {
      auto offsets = line.block_offsets(i);
      OTHER_ASSERT(   block_info.valid(block_id+1)
                   && offsets.x==first.x+block_id
                   && offsets.y==block_info[block_id].offset);
      const auto block = line.block(i);
      OTHER_ASSERT(partition.block_offsets(line.section,block)==vec(first.x+block_id,first.y+node));
      node += block_shape(line.section.shape(),block,partition.block_size).product();
      block_info[block_id].section = line.section;
      block_info[block_id].block = block;
      block_info[block_id].lock = spinlock_t();
      block_info[++block_id].offset = node;
    }
  OTHER_ASSERT(block_id==last.x-first.x);
  OTHER_ASSERT(node==last.y-first.y);
  const_cast_(this->block_info) = block_info;

  // Allocate space for all blocks as one huge array, and zero it to indicate losses.
  const_cast_(this->all_data) = aligned_buffer<Vector<super_t,2>>(last.y-first.y);
  memset(all_data.data(),0,sizeof(Vector<super_t,2>)*all_data.size());

  // Count the number of required contributions before all blocks are complete
  uint64_t count = 0;
  for (int b : range(blocks()))
    for (int i=0;i<4;i++)
      count += block_info[b].section.counts[i].sum()<9;
  OTHER_ASSERT(count<(1<<31));
  const_cast_(required_contributions) = count;
  missing_contributions = counter_t(count);
}

block_store_t::~block_store_t() {}

uint64_t block_store_t::memory_usage() const {
  return sizeof(block_store_t)+pentago::memory_usage(block_info)+pentago::memory_usage(all_data);
}

void block_store_t::accumulate(int local_id, RawArray<const Vector<super_t,2>> new_data) {
  OTHER_ASSERT(0<=local_id && local_id<blocks());
  const auto local_data = all_data.slice(block_info[local_id].offset,block_info[local_id+1].offset);
  OTHER_ASSERT(local_data.size()==new_data.size());
  {
    spin_t spin(block_info[local_id].lock);
    for (int i=0;i<local_data.size();i++) {
      local_data[i].x |= new_data[i].x;
      local_data[i].y |= new_data[i].y;
    }
  }
  --missing_contributions;
}

void block_store_t::set_complete() {
  OTHER_ASSERT(!missing_contributions);
  complete = true;
}

RawArray<const Vector<super_t,2>,4> block_store_t::get(section_t section, Vector<int,4> block) const {
  OTHER_ASSERT(complete);
  const auto offsets = partition->block_offsets(section,block);
  OTHER_ASSERT(first.x<=offsets.x && offsets.x<last.x);
  const auto shape = section.shape();
  OTHER_ASSERT(first.y<=offsets.y && offsets.y+shape.product()<=last.y);
  return RawArray<const Vector<super_t,2>,4>(block_shape(shape,block,partition->block_size),all_data.data()+offsets.y-first.y);
}

RawArray<const Vector<super_t,2>> block_store_t::get(int local_id) const {
  OTHER_ASSERT(complete);
  OTHER_ASSERT(0<=local_id && local_id<blocks());
  return all_data.slice(block_info[local_id].offset,block_info[local_id+1].offset);
}

}
}
