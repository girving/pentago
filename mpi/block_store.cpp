// Data structure keeping track of all blocks we own

#include <pentago/mpi/block_store.h>
#include <pentago/mpi/utility.h>
#include <pentago/count.h>
#include <pentago/utility/aligned.h>
#include <pentago/utility/memory.h>
#include <other/core/python/Class.h>
#include <other/core/utility/const_cast.h>
#include <other/core/utility/Log.h>
#include <boost/bind.hpp>
namespace pentago {
namespace mpi {

OTHER_DEFINE_TYPE(block_store_t)
using Log::cout;
using std::endl;

static const int block_size = 8;

block_store_t::block_store_t(const partition_t& partition, const int rank, Array<const line_t> lines)
  : partition(ref(partition))
  , rank(rank)
  , first(partition.rank_offsets(rank))
  , last(partition.rank_offsets(rank+1))
  , section_counts(partition.sections.size())
  , required_contributions(0) { // set below

  // Block size must be 8 (used in count_block_wins below)
  OTHER_ASSERT(partition.block_size==8);

  // Make sure 32-bit array indices suffice
  OTHER_ASSERT(last.y-first.y<(1u<<31));

  // Compute map from local block id to local offset
  int block_id = 0, node = 0;
  Array<block_info_t> block_info(last.x-first.x+1,false);
  block_info[0].offset = 0;
  for (auto line : lines)
    for (int i : range(line.length)) {
      auto offsets = line.block_offsets(i);
      OTHER_ASSERT(   block_info.valid(block_id+1)
                   && offsets.x==first.x+block_id
                   && offsets.y==first.y+block_info[block_id].offset);
      const auto block = line.block(i);
      OTHER_ASSERT(partition.block_offsets(line.section,block)==vec(first.x+block_id,first.y+node));
      node += block_shape(line.section.shape(),block,block_size).product();
      block_info[block_id].section = line.section;
      block_info[block_id].block = block;
      block_info[block_id].lock = spinlock_t();
      block_info[++block_id].offset = node;
    }
  OTHER_ASSERT((uint64_t)block_id==last.x-first.x);
  OTHER_ASSERT((uint64_t)node==last.y-first.y);
  const_cast_(this->block_info) = block_info;

  // Allocate space for all blocks as one huge array, and zero it to indicate losses.
  const_cast_(this->all_data) = aligned_buffer<Vector<super_t,2>>(last.y-first.y);
  memset(all_data.data(),0,sizeof(Vector<super_t,2>)*all_data.size());

  // Count the number of required contributions before all blocks are complete
  uint64_t total_count = 0;
  for (int b : range(blocks())) {
    int count = 0;
    for (int i=0;i<4;i++)
      count += block_info[b].section.counts[i].sum()<9;
    block_info[b].missing_contributions = count;
    total_count += count;
  }
  OTHER_ASSERT(total_count<(1u<<31));
  const_cast_(required_contributions) = total_count;
}

block_store_t::~block_store_t() {}

uint64_t block_store_t::memory_usage() const {
  return sizeof(block_store_t)+pentago::memory_usage(block_info)+pentago::memory_usage(all_data)+pentago::memory_usage(section_counts);
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
    if (!--block_info[local_id].missing_contributions)
      threads_schedule(CPU,boost::bind(&block_store_t::count_wins,this,local_id));
  }
}

RawArray<const Vector<super_t,2>,4> block_store_t::get(section_t section, Vector<int,4> block) const {
  const auto offsets = partition->block_offsets(section,block);
  const int local_id = offsets.x-first.x;
  OTHER_ASSERT((unsigned)local_id<(unsigned)blocks());
  OTHER_ASSERT(!block_info[local_id].missing_contributions);
  const auto shape = block_shape(section.shape(),block,block_size);
  OTHER_ASSERT(first.y<=offsets.y && offsets.y+shape.product()<=last.y);
  return RawArray<const Vector<super_t,2>,4>(shape,all_data.data()+offsets.y-first.y);
}

RawArray<const Vector<super_t,2>> block_store_t::get_flat(int local_id) const {
  OTHER_ASSERT((unsigned)local_id<(unsigned)blocks());
  OTHER_ASSERT(!block_info[local_id].missing_contributions);
  return all_data.slice(block_info[local_id].offset,block_info[local_id+1].offset);
}

RawArray<const Vector<super_t,2>,4> block_store_t::get(int local_id) const {
  const auto flat = get_flat(local_id);
  const auto shape = block_shape(block_info[local_id].section.shape(),block_info[local_id].block,block_size);
  OTHER_ASSERT(flat.size()==shape.product());
  return RawArray<const Vector<super_t,2>,4>(shape,flat.data());
}

void block_store_t::count_wins(int local_id) {
  // Prepare
  const auto flat_data = get_flat(local_id);
  const auto& info = block_info[local_id];

  // Count and store
  const int section_id = partition->section_id.get(info.section);
  const auto counts = count_block_wins(info.section,info.block,flat_data);
  spin_t spin(section_counts_lock);
  section_counts[section_id] += counts; 
}

Vector<uint64_t,3> count_block_wins(const section_t section, const Vector<int,4> block, RawArray<const Vector<super_t,2>> flat_data) {
  // Prepare
  const auto base = block_size*block;
  const auto shape = block_shape(section.shape(),block,block_size);
  const auto rmin0 = safe_rmin_slice(section.counts[0],base[0]+range(shape[0])),
             rmin1 = safe_rmin_slice(section.counts[1],base[1]+range(shape[1])),
             rmin2 = safe_rmin_slice(section.counts[2],base[2]+range(shape[2])),
             rmin3 = safe_rmin_slice(section.counts[3],base[3]+range(shape[3]));
  OTHER_ASSERT(shape.product()==flat_data.size());
  const RawArray<const Vector<super_t,2>,4> block_data(shape,flat_data.data());

  // Count
  Vector<uint64_t,3> counts;
  for (int i0=0;i0<shape[0];i0++)
    for (int i1=0;i1<shape[1];i1++)
      for (int i2=0;i2<shape[2];i2++)
        for (int i3=0;i3<shape[3];i3++)
          counts += Vector<uint64_t,3>(popcounts_over_stabilizers(quadrants(rmin0[i0],rmin1[i1],rmin2[i2],rmin3[i3]),block_data(i0,i1,i2,i3)));
  return counts;
}

}
}
using namespace other;
using namespace pentago::mpi;

void wrap_block_store() {
  typedef block_store_t Self;
  Class<Self>("block_store_t")
    .OTHER_METHOD(blocks)
    .OTHER_FIELD(section_counts)
    .OTHER_FIELD(all_data)
    ;
}
