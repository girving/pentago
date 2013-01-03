// Data structure keeping track of all blocks we own

#include <pentago/mpi/block_store.h>
#include <pentago/mpi/fast_compress.h>
#include <pentago/mpi/utility.h>
#include <pentago/count.h>
#include <pentago/utility/char_view.h>
#include <pentago/utility/index.h>
#include <pentago/utility/memory.h>
#include <other/core/array/Array4d.h>
#include <other/core/python/Class.h>
#include <other/core/random/Random.h>
#include <other/core/utility/const_cast.h>
#include <other/core/utility/Hasher.h>
#include <other/core/utility/Log.h>
#include <tr1/unordered_map>
namespace pentago {
namespace mpi {

OTHER_DEFINE_TYPE(block_store_t)
using std::tr1::unordered_map;
using std::make_pair;
using Log::cout;
using std::endl;

block_store_t::block_store_t(const partition_t& partition, const int rank, RawArray<const local_block_t> blocks, const int samples_per_section)
  : sections(partition.sections)
  , partition(ref(partition))
  , rank(rank)
  , total_nodes(0) // set below
  , section_counts(sections->sections.size())
  , required_contributions(0) // set below
#if PENTAGO_MPI_COMPRESS
  , store(blocks.size(),raw_max_fast_compressed_size)
#endif
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
  OTHER_ASSERT(next_flat_id==blocks.size());
  OTHER_ASSERT(total_count<(1u<<31));
  const_cast_(required_contributions) = total_count;
  // Make sure 32-bit array indices suffice
  OTHER_ASSERT(total_nodes<(1u<<31));
  const_cast_(this->total_nodes) = total_nodes;

#if !PENTAGO_MPI_COMPRESS
  // Allocate space for all blocks as one huge array, and zero it to indicate losses.
  const_cast_(this->all_data) = large_buffer<Vector<super_t,2>>(total_nodes,true);
#endif

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
  const_cast_(samples) = NestedArray<sample_t>(sample_counts,false);

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

block_store_t::~block_store_t() {}

uint64_t block_store_t::base_memory_usage() const {
  return sizeof(block_store_t)+pentago::memory_usage(block_infos)+pentago::memory_usage(section_counts);
}

uint64_t block_store_t::current_memory_usage() const {
  return base_memory_usage()
#if PENTAGO_MPI_COMPRESS
    +store.current_memory_usage();
#else
    +pentago::memory_usage(all_data);
#endif
}

uint64_t block_store_t::estimate_peak_memory_usage() const {
#if PENTAGO_MPI_COMPRESS
  return base_memory_usage()+sparse_store_t::estimate_peak_memory_usage(total_blocks(),snappy_compression_estimate*sizeof(Vector<super_t,2>)*(total_nodes));
#else
  return current_memory_usage();
#endif
}

const block_info_t& block_store_t::block_info(const local_id_t local_id) const {
  if (const auto* info = block_infos.get_pointer(local_id))
    return *info;
  const auto block = partition->rank_block(rank,local_id);
  die("block_store_t::block_info: invalid local id %d, block %s",local_id.id,str(block));
}

const block_info_t& block_store_t::block_info(const section_t section, const Vector<uint8_t,4> block) const {
  if (const auto* local_id = block_to_local_id.get_pointer(tuple(section,block)))
    return block_info(*local_id);
  die("block_store_t::block_info: unknown block: section %s, block %d,%d,%d,%d",str(section),block[0],block[1],block[2],block[3]);
}

void block_store_t::print_compression_stats(MPI_Comm comm) const {
#if PENTAGO_MPI_COMPRESS
  // We're going to compute the mean and variance of the compression ratio for a randomly chosen *byte*.
  // First integrate moments 0, 1, and 2.
  uint64_t uncompressed_ = 0;
  uint64_t compressed_ = 0,
           peak_compressed_ = 0;
  double sqr_compressed = 0,
         peak_sqr_compressed = 0;
  for (auto& info : block_infos) {
    const int flat_id = info.data.flat_id;
    const int k = 64*block_shape(info.data.section.shape(),info.data.block).product();
    uncompressed_ += k;
    compressed_ += store.size(flat_id);
    peak_compressed_ += store.peak_size(flat_id);
    sqr_compressed += sqr((double)store.size(flat_id))/k;
    peak_sqr_compressed += sqr((double)store.peak_size(flat_id))/k;
  }
  // Reduce to rank 0
  double numbers[5] = {(double)uncompressed_,(double)compressed_,(double)peak_compressed_,sqr_compressed,peak_sqr_compressed};
  const int rank = comm_rank(comm);
  CHECK(MPI_Reduce(rank?numbers:MPI_IN_PLACE,numbers,5,MPI_DOUBLE,MPI_SUM,0,comm));
  // Compute statistics
  if (!rank) {
    const double uncompressed = numbers[0],
                 compressed = numbers[1],
                 peak_compressed = numbers[2];
    sqr_compressed = numbers[3];
    peak_sqr_compressed = numbers[4];
    const double mean = compressed/uncompressed,
                 dev = sqrt(sqr_compressed/uncompressed-sqr(mean)),
                 peak_mean = peak_compressed/uncompressed,
                 peak_dev = sqrt(peak_sqr_compressed/uncompressed-sqr(peak_mean));
    cout << "compression ratio      = "<<mean<<" +- "<<dev<<endl; 
    cout << "peak compression ratio = "<<peak_mean<<" +- "<<peak_dev<<endl; 
  }
#endif
}

void block_store_t::accumulate(local_id_t local_id, uint8_t dimension, RawArray<Vector<super_t,2>> new_data) {
  OTHER_ASSERT(dimension<4);
  const auto& info = block_info(local_id);
  const int flat_id = info.flat_id;
  const event_t event = block_line_event(info.section,dimension,info.block);
#if !PENTAGO_MPI_COMPRESS
  const auto local_data = all_data.slice(info.nodes);
  OTHER_ASSERT(local_data.size()==new_data.size());
  {
    spin_t spin(info.lock);
    {
      thread_time_t time(accumulate_kind,event);
      for (int i=0;i<local_data.size();i++)
        local_data[i] |= new_data[i];
    }
    // If all contributions are in place, count and sample
    OTHER_ASSERT(info.missing_dimensions&1<<dimension);
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
  // If previous contributions exist, uncompress the old data and combine it with the new
  if (store.size(flat_id)) {
    const auto old_data = uncompress_and_get_flat(local_id,event,true);
    thread_time_t time(accumulate_kind,event);
    OTHER_ASSERT(new_data.size()==old_data.size());
    for (int i=0;i<new_data.size();i++)
      new_data[i] |= old_data[i];
  }
  // If all contributions are in place, count and sample
  OTHER_ASSERT(info.missing_dimensions&1<<dimension);
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
  store.compress_and_set(flat_id,new_data,event);
#endif
}

void block_store_t::assert_contains(section_t section, Vector<uint8_t,4> block) const {
  block_info(section,block);
}

event_t block_store_t::local_block_event(local_id_t local_id) const {
  const auto& info = block_info(local_id);
  return block_event(info.section,info.block);
}

event_t block_store_t::local_block_line_event(local_id_t local_id, uint8_t dimension) const {
  if (dimension>=4)
    die("block_store_t::local_block_line_event: local_id %d, blocks %d, dimension %d",local_id.id,total_blocks(),dimension);
  const auto& info = block_info(local_id);
  return block_line_event(info.section,dimension,info.block);
}

event_t block_store_t::local_block_lines_event(local_id_t local_id, dimensions_t dimensions) const {
  const auto& info = block_info(local_id);
  return block_lines_event(info.section,dimensions,info.block);
}

#if PENTAGO_MPI_COMPRESS

Array<Vector<super_t,2>,4> block_store_t::uncompress_and_get(section_t section, Vector<uint8_t,4> block, event_t event) const {
  if (const auto* local_id = block_to_local_id.get_pointer(tuple(section,block)))
    return uncompress_and_get(*local_id,event);
  die("block_store_t::uncompress_and_get: unknown block: section %s, block %d,%d,%d,%d",str(section),block[0],block[1],block[2],block[3]);
}

Array<Vector<super_t,2>,4> block_store_t::uncompress_and_get(local_id_t local_id, event_t event) const {
  const auto flat = uncompress_and_get_flat(local_id,event);
  const auto& info = block_info(local_id);
  const auto shape = block_shape(info.section.shape(),info.block);
  return Array<Vector<super_t,2>,4>(shape,flat.data(),flat.borrow_owner());
}

Array<Vector<super_t,2>> block_store_t::uncompress_and_get_flat(local_id_t local_id, event_t event, bool allow_incomplete) const {
  const auto compressed = get_compressed(local_id,allow_incomplete);
  const auto& info = block_info(local_id);
  const int flat_size = block_shape(info.section.shape(),info.block).product();
  // Uncompress into a temporary array.  For sake of simplicity, we'll hope that malloc manages to avoid fragmentation.
  // In the future, we may want to keep a list of temporary blocks around to eliminate fragmentation entirely.
  const auto flat = large_buffer<Vector<super_t,2>>(flat_size,false);
  fast_uncompress(compressed,flat,event);
  return flat;
}

RawArray<const uint8_t> block_store_t::get_compressed(local_id_t local_id, bool allow_incomplete) const {
  const auto& info = block_info(local_id);
  if (!allow_incomplete)
    OTHER_ASSERT(!info.missing_dimensions);
  return store.current_buffer(info.flat_id);
}

#else

RawArray<const Vector<super_t,2>,4> block_store_t::get_raw(section_t section, Vector<uint8_t,4> block) const {
  const auto& info = block_info(section,block);
  OTHER_ASSERT(!info.missing_dimensions);
  const auto shape = block_shape(section.shape(),block);
  OTHER_ASSERT(0<=info.nodes.lo && info.nodes.hi<=all_data.size());
  return RawArray<const Vector<super_t,2>,4>(shape,all_data.data()+info.nodes.lo);
}

RawArray<const Vector<super_t,2>> block_store_t::get_raw_flat(local_id_t local_id) const {
  const auto& info = block_info.get(local_id);
  OTHER_ASSERT(!info.missing_dimensions);
  return all_data.slice(info.nodes);
}

RawArray<const Vector<super_t,2>,4> block_store_t::get_raw(local_id_t local_id) const {
  const auto& info = block_info.get(local_id);
  OTHER_ASSERT(!info.missing_dimensions);
  const auto flat = all_data.slice(info.nodes);
  const auto shape = block_shape(info.section.shape(),info.block);
  OTHER_ASSERT(flat.size()==shape.product());
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
using namespace pentago::mpi;

void wrap_block_store() {
  typedef block_store_t Self;
  Class<Self>("block_store_t")
    .OTHER_METHOD(total_blocks)
    .OTHER_FIELD(total_nodes)
    .OTHER_FIELD(section_counts)
    ;
}
