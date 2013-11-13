// Data structure keeping track of all blocks we own

#include <pentago/end/block_store.h>
#include <pentago/base/count.h>
#include <pentago/end/fast_compress.h>
#include <pentago/end/blocks.h>
#include <pentago/utility/char_view.h>
#include <pentago/utility/debug.h>
#include <pentago/utility/index.h>
#include <pentago/utility/memory.h>
#include <geode/array/Array4d.h>
#include <geode/python/Class.h>
#include <geode/random/Random.h>
#include <geode/utility/const_cast.h>
#include <geode/utility/Hasher.h>
#include <geode/utility/Log.h>
#include <geode/utility/tr1.h>
namespace pentago {
namespace end {

GEODE_DEFINE_TYPE(readable_block_store_t)
GEODE_DEFINE_TYPE(accumulating_block_store_t)
GEODE_DEFINE_TYPE(restart_block_store_t)
using std::make_pair;
using Log::cout;
using std::endl;

readable_block_store_t::readable_block_store_t(const block_partition_t& partition, const int rank, RawArray<const local_block_t> blocks, compacting_store_t& store)
  : sections(partition.sections)
  , partition(ref(partition))
  , rank(rank)
  , total_nodes(0) // set below
  , required_contributions(0) // set below
  , store(store,blocks.size())
{
  uint64_t total_nodes = 0;
  uint64_t total_count = 0;
  int next_flat_id = 0;
  for (const auto& block : blocks) {
    block_info_t info;
    info.section = block.section;
    info.block = block.block;
    info.flat_id = next_flat_id++;
    info.lock = spinlock_t();
    const int nodes = block_shape(info.section.shape(),info.block).product();
#if !PENTAGO_MPI_COMPRESS
    info.nodes = int(total_nodes)+range(nodes);
#endif
    total_nodes += nodes;
    // Count missing dimensions
    uint8_t dimensions = 0;
    for (int i=0;i<4;i++)
      if (info.section.counts[i].sum()<9)
        dimensions |= 1<<i;
    info.missing_dimensions = dimensions;
    total_count += popcount((uint16_t)dimensions);
    // Map local id to block info
    const_cast_(block_infos).set(block.local_id,info);
    const_cast_(block_to_local_id).set(tuple(info.section,info.block),block.local_id);
  }
  GEODE_ASSERT(next_flat_id==blocks.size());
  GEODE_ASSERT(total_count<(1u<<31));
  const_cast_(required_contributions) = total_count;
  // Make sure 32-bit array indices suffice
  GEODE_ASSERT(total_nodes<(1u<<31));
  const_cast_(this->total_nodes) = total_nodes;

#if !PENTAGO_MPI_COMPRESS
  // Allocate space for all blocks as one huge array, and zero it to indicate losses.
  const_cast_(this->all_data) = large_buffer<Vector<super_t,2>>(total_nodes,true);
#endif
}

accumulating_block_store_t::accumulating_block_store_t(const block_partition_t& partition, const int rank, RawArray<const local_block_t> blocks, const int samples_per_section, compacting_store_t& store)
  : Base(partition,rank,blocks,store)
  , section_counts(sections->sections.size()) {
  // Count random samples in each block
  Array<int> sample_counts(total_blocks());
  for (const auto section : sections->sections) {
    const auto shape = section.shape();
    const auto rmin = vec(rotation_minimal_quadrants(section.counts[0]).x,
                          rotation_minimal_quadrants(section.counts[1]).x,
                          rotation_minimal_quadrants(section.counts[2]).x,
                          rotation_minimal_quadrants(section.counts[3]).x);
    const auto random = new_<Random>(hash(section));
    for (int i=0;i<samples_per_section;i++) {
      const auto index = random->uniform(Vector<int,4>(),shape);
      const auto block = Vector<uint8_t,4>(index/block_size);
      if (const auto* local_id = block_to_local_id.get_pointer(tuple(section,block)))
        sample_counts[block_info(*local_id).flat_id]++;
    }
  }
  const_cast_(samples) = Nested<sample_t>(sample_counts,false);

  // Prepare to collect random samples from each block.  Note that this duplicates the loop from above.
  for (const auto section : sections->sections) {
    const auto shape = section.shape();
    const auto rmin = vec(rotation_minimal_quadrants(section.counts[0]).x,
                          rotation_minimal_quadrants(section.counts[1]).x,
                          rotation_minimal_quadrants(section.counts[2]).x,
                          rotation_minimal_quadrants(section.counts[3]).x);
    const auto random = new_<Random>(hash(section));
    for (int i=0;i<samples_per_section;i++) {
      const auto index = random->uniform(Vector<int,4>(),shape);
      const auto block = Vector<uint8_t,4>(index/block_size);
      if (auto* local_id = block_to_local_id.get_pointer(tuple(section,block))) {
        const int flat_id = block_info(*local_id).flat_id;
        auto& sample = samples[flat_id][--sample_counts[flat_id]];
        sample.board = quadrants(rmin[0][index[0]],
                                 rmin[1][index[1]],
                                 rmin[2][index[2]],
                                 rmin[3][index[3]]);
        sample.index = pentago::index(block_shape(shape,block),index-block_size*Vector<int,4>(block));
      }
    }
  }
}

restart_block_store_t::restart_block_store_t(const block_partition_t& partition, const int rank, RawArray<const local_block_t> blocks, compacting_store_t& store)
  : Base(partition,rank,blocks,store) {}

readable_block_store_t::~readable_block_store_t() {}
accumulating_block_store_t::~accumulating_block_store_t() {}
restart_block_store_t::~restart_block_store_t() {}

uint64_t readable_block_store_t::base_memory_usage() const {
  return sizeof(readable_block_store_t)+pentago::memory_usage(block_infos);
}

uint64_t accumulating_block_store_t::base_memory_usage() const {
  return Base::base_memory_usage()+pentago::memory_usage(section_counts);
}

const block_info_t& readable_block_store_t::block_info(const local_id_t local_id) const {
  if (const auto* info = block_infos.get_pointer(local_id))
    return *info;
  const auto block = partition->rank_block(rank,local_id);
  die("readable_block_store_t::block_info: invalid local id %d, block %s",local_id.id,str(block));
}

const block_info_t& readable_block_store_t::block_info(const section_t section, const Vector<uint8_t,4> block) const {
  if (const auto* local_id = block_to_local_id.get_pointer(tuple(section,block)))
    return block_info(*local_id);
  die("readable_block_store_t::block_info: unknown block: section %s, block %d,%d,%d,%d",str(section),block[0],block[1],block[2],block[3]);
}

void readable_block_store_t::print_compression_stats(const reduction_t<double,sum_op>& reduce_sum) const {
#if PENTAGO_MPI_COMPRESS
  // We're going to compute the mean and variance of the compression ratio for a randomly chosen *byte*.
  // First integrate moments 0, 1, and 2.
  uint64_t uncompressed_ = 0;
  uint64_t compressed_ = 0;
  double sqr_compressed = 0;
  for (auto& info : block_infos) {
    const int flat_id = info.data().flat_id;
    const int k = 64*block_shape(info.data().section.shape(),info.data().block).product();
    uncompressed_ += k;
    const int size = store.get_frozen(flat_id).size();
    compressed_ += size;
    sqr_compressed += sqr((double)size)/k;
  }
  // Reduce to rank 0
  double numbers[3] = {(double)uncompressed_,(double)compressed_,sqr_compressed};
  const bool root = !reduce_sum || reduce_sum(RawArray<double>(3,numbers));
  // Compute statistics
  if (root) {
    const double uncompressed = numbers[0],
                 compressed = numbers[1],
                 sqr_compressed = numbers[2];
    const double mean = compressed/uncompressed,
                 dev = sqrt(sqr_compressed/uncompressed-sqr(mean));
    cout << "compression ratio = "<<mean<<" +- "<<dev<<endl;
  }
#endif
}

void accumulating_block_store_t::accumulate(local_id_t local_id, uint8_t dimension, RawArray<Vector<super_t,2>> new_data) {
  GEODE_ASSERT(dimension<4);
  const auto& info = block_info(local_id);
  const int flat_id = info.flat_id;
  const event_t event = block_line_event(info.section,dimension,info.block);
#if !PENTAGO_MPI_COMPRESS
  const auto local_data = all_data.slice(info.nodes);
  GEODE_ASSERT(local_data.size()==new_data.size());
  {
    spin_t spin(info.lock);
    {
      thread_time_t time(accumulate_kind,event);
      for (int i=0;i<local_data.size();i++)
        local_data[i] |= new_data[i];
    }
    // If all contributions are in place, count and sample
    GEODE_ASSERT(info.missing_dimensions&1<<dimension);
    info.missing_dimensions &= ~(1<<dimension);
    if (!info.missing_dimensions) {
      thread_time_t time(count_kind,event);
      for (auto& sample : samples[flat_id])
        sample.wins = local_data[sample.index];
      const auto counts = count_block_wins(info.section,info.block,local_data);
      const int section_id = sections->section_id.get(info.section);
      spin_t spin(section_counts_lock);
      section_counts[section_id] += counts;
    }
  }
#else // PENTAGO_MPI_COMPRESS
  spin_t spin(info.lock);
  compacting_store_t::lock_t alock(store,flat_id);
  // If previous contributions exist, uncompress the old data and combine it with the new
  {
    const auto old_compressed = alock.get(); 
    if (old_compressed.size()) {
      const auto old_data = local_fast_uncompress(old_compressed,event);      
      thread_time_t time(accumulate_kind,event);
      GEODE_ASSERT(new_data.size()==old_data.size());
      for (int i=0;i<new_data.size();i++)
        new_data[i] |= old_data[i];
    }
  }
  // If all contributions are in place, count and sample
  GEODE_ASSERT(info.missing_dimensions&1<<dimension);
  info.missing_dimensions &= ~(1<<dimension);
  if (!info.missing_dimensions) {
    thread_time_t time(count_kind,event);
    for (auto& sample : samples[flat_id])
      sample.wins = new_data[sample.index];
    const auto counts = count_block_wins(info.section,info.block,new_data);
    const int section_id = sections->section_id.get(info.section);
    spin_t spin(section_counts_lock);
    section_counts[section_id] += counts;
  }
  // Compress data into place
  const auto new_compressed = local_fast_compress(new_data,event);
  thread_time_t time(compacting_kind,event);
  alock.set(new_compressed);
#endif
}

void restart_block_store_t::set(local_id_t local_id, RawArray<Vector<super_t,2>> new_data) {
  const auto& info = block_info(local_id);
  const int flat_id = info.flat_id;
  const event_t event = block_event(info.section,info.block);
#if !PENTAGO_MPI_COMPRESS
  const auto local_data = all_data.slice(info.nodes);
  GEODE_ASSERT(local_data.size()==new_data.size());
  spin_t spin(info.lock);
  GEODE_ASSERT(info.missing_dimensions);
  info.missing_dimensions = 0;
  local_data = new_data;
#else // PENTAGO_MPI_COMPRESS
  spin_t spin(info.lock);
  GEODE_ASSERT(info.missing_dimensions);
  info.missing_dimensions = 0;
  compacting_store_t::lock_t alock(store,flat_id);
  // Compress data into place
  const auto new_compressed = local_fast_compress(new_data,event);
  thread_time_t time(compacting_kind,event);
  alock.set(new_compressed);
#endif
}

void readable_block_store_t::assert_contains(section_t section, Vector<uint8_t,4> block) const {
  block_info(section,block);
}

event_t readable_block_store_t::local_block_event(local_id_t local_id) const {
  const auto& info = block_info(local_id);
  return block_event(info.section,info.block);
}

event_t readable_block_store_t::local_block_line_event(local_id_t local_id, uint8_t dimension) const {
  if (dimension>=4)
    die("readable_block_store_t::local_block_line_event: local_id %d, blocks %d, dimension %d",local_id.id,total_blocks(),dimension);
  const auto& info = block_info(local_id);
  return block_line_event(info.section,dimension,info.block);
}

event_t readable_block_store_t::local_block_lines_event(local_id_t local_id, dimensions_t dimensions) const {
  const auto& info = block_info(local_id);
  return block_lines_event(info.section,dimensions,info.block);
}

#if PENTAGO_MPI_COMPRESS

RawArray<Vector<super_t,2>,4> readable_block_store_t::uncompress_and_get(section_t section, Vector<uint8_t,4> block, event_t event) const {
  if (const auto* local_id = block_to_local_id.get_pointer(tuple(section,block)))
    return uncompress_and_get(*local_id,event);
  die("readable_block_store_t::uncompress_and_get: unknown block: section %s, block %d,%d,%d,%d",str(section),block[0],block[1],block[2],block[3]);
}

RawArray<Vector<super_t,2>,4> readable_block_store_t::uncompress_and_get(local_id_t local_id, event_t event) const {
  const auto flat = uncompress_and_get_flat(local_id,event);
  const auto& info = block_info(local_id);
  const auto shape = block_shape(info.section.shape(),info.block);
  return RawArray<Vector<super_t,2>,4>(shape,flat.data());
}

RawArray<Vector<super_t,2>> readable_block_store_t::uncompress_and_get_flat(local_id_t local_id, event_t event) const {
  return local_fast_uncompress(get_compressed(local_id),event);
}

RawArray<const uint8_t> readable_block_store_t::get_compressed(local_id_t local_id) const {
  const auto& info = block_info(local_id);
  GEODE_ASSERT(!info.missing_dimensions);
  return store.get_frozen(info.flat_id);
}

#else

RawArray<const Vector<super_t,2>,4> readable_block_store_t::get_raw(section_t section, Vector<uint8_t,4> block) const {
  const auto& info = block_info(section,block);
  GEODE_ASSERT(!info.missing_dimensions);
  const auto shape = block_shape(section.shape(),block);
  GEODE_ASSERT(0<=info.nodes.lo && info.nodes.hi<=all_data.size());
  return RawArray<const Vector<super_t,2>,4>(shape,all_data.data()+info.nodes.lo);
}

RawArray<const Vector<super_t,2>> readable_block_store_t::get_raw_flat(local_id_t local_id) const {
  const auto& info = block_info.get(local_id);
  GEODE_ASSERT(!info.missing_dimensions);
  return all_data.slice(info.nodes);
}

RawArray<const Vector<super_t,2>,4> readable_block_store_t::get_raw(local_id_t local_id) const {
  const auto& info = block_info.get(local_id);
  GEODE_ASSERT(!info.missing_dimensions);
  const auto flat = all_data.slice(info.nodes);
  const auto shape = block_shape(info.section.shape(),info.block);
  GEODE_ASSERT(flat.size()==shape.product());
  return RawArray<const Vector<super_t,2>,4>(shape,flat.data());
}

#endif

Vector<uint64_t,3> count_block_wins(const section_t section, const Vector<uint8_t,4> block, RawArray<const Vector<super_t,2>> flat_data) {
  // Prepare
  const auto base = block_size*Vector<int,4>(block);
  const auto shape = block_shape(section.shape(),block);
  const auto rmin0 = safe_rmin_slice(section.counts[0],base[0]+range(shape[0])),
             rmin1 = safe_rmin_slice(section.counts[1],base[1]+range(shape[1])),
             rmin2 = safe_rmin_slice(section.counts[2],base[2]+range(shape[2])),
             rmin3 = safe_rmin_slice(section.counts[3],base[3]+range(shape[3]));
  GEODE_ASSERT(shape.product()==flat_data.size());
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
using namespace pentago::end;

void wrap_block_store() {
  {
    typedef readable_block_store_t Self;
    Class<Self>("readable_block_store_t")
      .GEODE_METHOD(total_blocks)
      .GEODE_FIELD(total_nodes)
      ;
  } {
    typedef accumulating_block_store_t Self;
    Class<Self>("accumulating_block_store_t")
      .GEODE_FIELD(section_counts)
      ;
  } {
    typedef restart_block_store_t Self;
    Class<Self>("restart_block_store_t")
      ;
  }
}
