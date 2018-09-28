// Data structure keeping track of all blocks we own
//
// block_store_t organizes all the block data owned by a given process.  This consists
// either of read only, finalized results (readable_block_store_t), or possibly incomplete
// writable results.  Note that data for a given block arrives in four pieces, once for
// each quadrant of board, and must be bitwise or'ed together before it is complete.
#pragma once

#include "pentago/end/partition.h"
#include "pentago/end/reduction.h"
#include "pentago/base/superscore.h"
#include "pentago/end/config.h"
#include "pentago/utility/counter.h"
#include "pentago/utility/spinlock.h"
#include "pentago/utility/nested.h"
#if PENTAGO_MPI_COMPRESS
#include "pentago/end/compacting_store.h"
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

using std::make_shared;

struct block_info_t {
  section_t section;
  Vector<uint8_t,4> block;
  int flat_id; // Contiguous local id (probably different from local id)
  mutable uint8_t missing_dimensions; // Which incoming dimension contributions are needed to complete this block
  mutable spinlock_t lock; // Used by accumulate
};
static_assert(sizeof(block_info_t)==32-8*PENTAGO_MPI_COMPRESS,"");

// A readable block store, not including counts and samples
class readable_block_store_t : public boost::noncopyable {
public:
  const shared_ptr<const sections_t> sections;
  const shared_ptr<const block_partition_t> partition;
  const int rank;
  const int total_nodes; // Total number of nodes
  const unordered_map<local_id_t,block_info_t> block_infos; // Local id to info about each block we own
  const unordered_map<tuple<section_t,Vector<uint8_t,4>>,local_id_t, // Block to local id
                      boost::hash<tuple<section_t,Vector<uint8_t,4>>>> block_to_local_id;
  const int required_contributions;
  compacting_store_t::group_t store; // Underlying data storage

  readable_block_store_t(const shared_ptr<const block_partition_t>& partition, const int rank,
                         RawArray<const local_block_t> blocks,
                         const shared_ptr<compacting_store_t>& store);
  virtual ~readable_block_store_t();

  // Number of blocks
  int total_blocks() const {
    return block_infos.size();
  }

  // Compute memory usage ignoring the store
  virtual uint64_t base_memory_usage() const;

  // Print statistics about block compression.
  void print_compression_stats(const reduction_t<double,sum_op>& reduce_sum) const;

  // Verify that we own the given block
  void assert_contains(section_t section, Vector<uint8_t,4> block) const;

  // Generate events for the given local block
  event_t local_block_event(local_id_t local_id) const;
  event_t local_block_line_event(local_id_t local_id, uint8_t dimension) const;
  event_t local_block_lines_event(local_id_t local_id, dimensions_t dimensions) const;

  // Access the data for a completed block, either by (section,block) or local block id.
  // In uncompressed mode, these are O(1) and return views into all_data.  In compressed mode they must uncompress
  // first, requiring O(n) time, and return new mutable buffers.  To make sure all callers know about these differences, we
  // give the different versions different names.
#if PENTAGO_MPI_COMPRESS
  // All uncompressed_and_get versions return views into a temporary, thread local buffer (the one used by local_fast_uncompress).
  RawArray<Vector<super_t,2>,4> uncompress_and_get(section_t section, Vector<uint8_t,4> block, event_t event) const;
  RawArray<Vector<super_t,2>,4> uncompress_and_get(local_id_t local_id, event_t event) const;
  RawArray<Vector<super_t,2>> uncompress_and_get_flat(local_id_t local_id, event_t event) const;
  RawArray<const uint8_t> get_compressed(local_id_t local_id) const;
#else
  RawArray<const Vector<super_t,2>,4> get_raw(section_t section, Vector<uint8_t,4> block) const;
  RawArray<const Vector<super_t,2>,4> get_raw(local_id_t local_id) const;
  RawArray<const Vector<super_t,2>> get_raw_flat(local_id_t local_id) const;
#endif

  // Look up info for a block
  const block_info_t& block_info(const local_id_t local_id) const;
  const block_info_t& block_info(const section_t section, const Vector<uint8_t,4> block) const;
};

// Accumulating block store, including counts and samples
class accumulating_block_store_t : public readable_block_store_t {
public:
  typedef readable_block_store_t Base;

  const Array<Vector<uint64_t,3>> section_counts; // Win/(win-or-tie)/total counts for each section (player to move first)
  spinlock_t section_counts_lock;

  // Space for sparse samples (filled in as blocks complete).  These are stored in native
  // block store format, and must be transformed before being written to disk.
  struct sample_t {
    board_t board;
    int index; // Index into flattened block data array
    Vector<super_t,2> wins;
  };
  const Nested<sample_t> samples;

  accumulating_block_store_t(const shared_ptr<const block_partition_t>& partition, const int rank,
                             RawArray<const local_block_t> blocks, const int samples_per_section,
                             const shared_ptr<compacting_store_t>& store);
  ~accumulating_block_store_t();

  // Compute memory usage ignoring the store
  virtual uint64_t base_memory_usage() const;

  // Accumulate new data into a block and count if the block is complete.  new_data is destroyed.  This function is thread safe.
  void accumulate(local_id_t local_id, uint8_t dimension, RawArray<Vector<super_t,2>> new_data);
};

// Nonaccumulating writable block store for use with restarts
class restart_block_store_t : public readable_block_store_t {
public:
  typedef readable_block_store_t Base;

  restart_block_store_t(const shared_ptr<const block_partition_t>& partition, const int rank,
                        RawArray<const local_block_t> blocks,
                        const shared_ptr<compacting_store_t>& store);
  ~restart_block_store_t();

  // Set completed block data.  new_data is destroyed.  This function is thread safe.
  void set(local_id_t local_id, RawArray<Vector<super_t,2>> new_data);
};

// Convenience factory routine
static inline shared_ptr<accumulating_block_store_t>
make_block_store(const shared_ptr<const block_partition_t>& partition, const int rank,
                 const int samples_per_section, const shared_ptr<compacting_store_t>& store) {
  return make_shared<accumulating_block_store_t>(
      partition, rank, partition->rank_blocks(rank), samples_per_section, store);
}

// The kernel of count_wins factored out for use elsewhere
Vector<uint64_t,3> count_block_wins(const section_t section, const Vector<uint8_t,4> block,
                                    RawArray<const Vector<super_t,2>> data);

}
}
