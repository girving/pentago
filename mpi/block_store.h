// Data structure keeping track of all blocks we own
#pragma once

#include <pentago/mpi/config.h>
#include <pentago/mpi/partition.h>
#include <pentago/superscore.h>
#include <pentago/utility/counter.h>
#include <pentago/utility/spinlock.h>
#include <other/core/array/NestedArray.h>
#ifdef PENTAGO_MPI_COMPRESS
#include <pentago/mpi/sparse_store.h>
#endif
namespace pentago {
namespace mpi {

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
  Vector<int,4> block;
#if !PENTAGO_MPI_COMPRESS
  int offset; // Local offset into all_data
#endif
  mutable int missing_contributions; // How many incoming contributions are needed to complete this block
  mutable spinlock_t lock; // Used by accumulate
};
BOOST_STATIC_ASSERT(sizeof(block_info_t)<=40);

class block_store_t : public Object {
public:
  OTHER_DECLARE_TYPE

  const Ref<const partition_t> partition;
  const int rank;
  const Vector<uint64_t,2> first, last; // Ranges of block and node ids
  const Array<const block_info_t> block_info; // Information about each block we own, plus one sentinel
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
  block_store_t(const partition_t& partition, const int rank, const int samples_per_section, Array<const line_t> lines);
public:
  ~block_store_t();

  // Number of blocks
  int blocks() const {
    return block_info.size()-1;
  }

  // Total number of nodes
  int nodes() const {
    return last.y-first.y;
  }

  // Estimate current memory usage
  uint64_t current_memory_usage() const;

  // Estimate peak memory usage.  In compressed mode, this is based on a hard coded guess as to how well snappy compresses.
  uint64_t estimate_peak_memory_usage() const;

  // Print statistics about block compression
  void print_compression_stats() const;

  // Verify that we own the given block
  void assert_contains(section_t section, Vector<int,4> block) const;

  // Accumulate new data into a block and count if the block is complete.  new_data is destroyed.  This function is thread safe.
  void accumulate(int local_id, RawArray<Vector<super_t,2>> new_data);

  // Access the data for a completed block, either by (section,block) or local block id.
  // In uncompressed mode, these are O(1) and return views into all_data.  In compressed mode they must uncompress
  // first, requiring O(n) time, and return new mutable buffers.  To make sure all callers know about these differences, we
  // give the different versions different names.
#if PENTAGO_MPI_COMPRESS
  Array<Vector<super_t,2>,4> uncompress_and_get(section_t section, Vector<int,4> block) const;
  Array<Vector<super_t,2>,4> uncompress_and_get(int local_id) const;
  Array<Vector<super_t,2>> uncompress_and_get_flat(int local_id, bool allow_incomplete=false) const; // allow_incomplete for internal use only
  RawArray<const char> get_compressed(int local_id, bool allow_incomplete=false) const;
#else
  RawArray<const Vector<super_t,2>,4> get_raw(section_t section, Vector<int,4> block) const;
  RawArray<const Vector<super_t,2>,4> get_raw(int local_id) const;
  RawArray<const Vector<super_t,2>> get_raw_flat(int local_id) const;
#endif

private:
  uint64_t base_memory_usage() const;
};

// The kernel of count_wins factored out for use elsewhere
Vector<uint64_t,3> count_block_wins(const section_t section, const Vector<int,4> block, RawArray<const Vector<super_t,2>> data);

}
}
