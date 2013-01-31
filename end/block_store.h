// Data structure keeping track of all blocks we own
#pragma once

#include <pentago/end/partition.h>
#include <pentago/end/reduction.h>
#include <pentago/base/superscore.h>
#include <pentago/end/config.h>
#include <pentago/utility/counter.h>
#include <pentago/utility/spinlock.h>
#include <other/core/array/NestedArray.h>
#if PENTAGO_MPI_COMPRESS
#include <pentago/end/sparse_store.h>
#endif
namespace pentago {
namespace end {

/* Important note:
 *
 * In order to support MPI_Accumulate, which disallows user defined reduction operations, we store
 * win data in a different format from that used elsewhere (on disk or in the out-of-core solver).
 * The format elsewhere is
 *
 *   x,y = black-wins, white-wins
 *
 * If it's black's turn, the required reduction operation is
 *
 *   x,y = x0|x1, y0&y1
 *
 * Instead, we use the format
 *
 *   x,y = we-win, we-win-or-tie
 *
 * where "we" is the player to move.  The reduction becomes pure bitwise or, and all is well.
 *
 * Update: Due to memory constraints and the lack of decent RMA synchronization, we can't
 * use MPI_Accumulate after all.  However, looking over the out-of-core endgame code, this
 * format does seem the best fit for the computation, so I'm still going to use it.
 */

struct block_info_t {
  section_t section;
  Vector<uint8_t,4> block;
  int flat_id; // Contiguous local id (probably different from local id)
#if !PENTAGO_MPI_COMPRESS
  Range<int> nodes; // Our piece of all_data
#endif
  mutable uint8_t missing_dimensions; // Which incoming dimension contributions are needed to complete this block
  mutable spinlock_t lock; // Used by accumulate
};
BOOST_STATIC_ASSERT(sizeof(block_info_t)==32-8*PENTAGO_MPI_COMPRESS);

class block_store_t : public Object {
public:
  OTHER_DECLARE_TYPE(OTHER_EXPORT)

  const Ref<const sections_t> sections;
  const Ref<const partition_t> partition;
  const int rank;
  const int total_nodes; // Total number of nodes
  const Hashtable<local_id_t,block_info_t> block_infos; // Map from local id to information about each block we own
  const Hashtable<Tuple<section_t,Vector<uint8_t,4>>,local_id_t> block_to_local_id; // Map from block to local id
  const Array<Vector<uint64_t,3>> section_counts; // Win/(win-or-tie)/total counts for each section 
  const int required_contributions;
  spinlock_t section_counts_lock;

#if PENTAGO_MPI_COMPRESS
  sparse_store_t store;
#else
  const Array<Vector<super_t,2>> all_data;
#endif

  // Space for sparse samples (filled in as blocks complete).  These are stored in native
  // block_store_t format, and must be transformed before being written to disk.
  struct sample_t {
    board_t board;
    int index; // Index into flattened block data array
    Vector<super_t,2> wins;
  };
  const NestedArray<sample_t> samples;

private:
  OTHER_EXPORT block_store_t(const partition_t& partition, const int rank, RawArray<const local_block_t> blocks, const int samples_per_section);
public:
  ~block_store_t();

  // Number of blocks
  int total_blocks() const {
    return block_infos.size();
  }

  // Estimate current memory usage
  uint64_t current_memory_usage() const;

  // Estimate peak memory usage.  In compressed mode, this is based on a hard coded guess as to how well snappy compresses.
  OTHER_EXPORT uint64_t estimate_peak_memory_usage() const;

  // Print statistics about block compression.
  OTHER_EXPORT void print_compression_stats(const reduction_t<double,sum_op>& reduce_sum) const;

  // Verify that we own the given block
  void assert_contains(section_t section, Vector<uint8_t,4> block) const;

  // Generate events for the given local block
  OTHER_EXPORT event_t local_block_event(local_id_t local_id) const;
  OTHER_EXPORT event_t local_block_line_event(local_id_t local_id, uint8_t dimension) const;
  OTHER_EXPORT event_t local_block_lines_event(local_id_t local_id, dimensions_t dimensions) const;

  // Accumulate new data into a block and count if the block is complete.  new_data is destroyed.  This function is thread safe.
  OTHER_EXPORT void accumulate(local_id_t local_id, uint8_t dimension, RawArray<Vector<super_t,2>> new_data);

  // Access the data for a completed block, either by (section,block) or local block id.
  // In uncompressed mode, these are O(1) and return views into all_data.  In compressed mode they must uncompress
  // first, requiring O(n) time, and return new mutable buffers.  To make sure all callers know about these differences, we
  // give the different versions different names.
#if PENTAGO_MPI_COMPRESS
  // All uncompressed_and_get versions return views into a temporary, thread local buffer (the one used by local_fast_uncompress).
  RawArray<Vector<super_t,2>,4> uncompress_and_get(section_t section, Vector<uint8_t,4> block, event_t event) const;
  RawArray<Vector<super_t,2>,4> uncompress_and_get(local_id_t local_id, event_t event) const;
  OTHER_EXPORT RawArray<Vector<super_t,2>> uncompress_and_get_flat(local_id_t local_id, event_t event, bool allow_incomplete=false) const; // allow_incomplete for internal use only
  OTHER_EXPORT RawArray<const uint8_t> get_compressed(local_id_t local_id, bool allow_incomplete=false) const;
#else
  RawArray<const Vector<super_t,2>,4> get_raw(section_t section, Vector<uint8_t,4> block) const;
  RawArray<const Vector<super_t,2>,4> get_raw(local_id_t local_id) const;
  RawArray<const Vector<super_t,2>> get_raw_flat(local_id_t local_id) const;
#endif

  // Look up info for a block
  const block_info_t& block_info(const local_id_t local_id) const;
  const block_info_t& block_info(const section_t section, const Vector<uint8_t,4> block) const;

private:
  uint64_t base_memory_usage() const;
};

// Convenience factory routine
static inline Ref<block_store_t> make_block_store(const partition_t& partition, const int rank, const int samples_per_section) {
  return new_<block_store_t>(partition,rank,partition.rank_blocks(rank),samples_per_section);
}

// The kernel of count_wins factored out for use elsewhere
Vector<uint64_t,3> count_block_wins(const section_t section, const Vector<uint8_t,4> block, RawArray<const Vector<super_t,2>> data);

}
}
