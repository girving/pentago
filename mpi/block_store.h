// Data structure keeping track of all blocks we own
#pragma once

#include <pentago/mpi/partition.h>
#include <pentago/superscore.h>
#include <pentago/utility/counter.h>
#include <pentago/utility/spinlock.h>
#include <mpi.h>
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
  int offset; // Local offset into all_data
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
  const Array<Vector<super_t,2>> all_data;
  const Array<Vector<uint64_t,3>> section_counts; // Win/(win-or-tie)/total counts for each section 
  const int required_contributions;
  static const bool compressed = false; // For now, we're always uncompressed
private:
  spinlock_t section_counts_lock;

  // Allocate all blocks initialized to zero (loss), and initialize a window for one-sided accumulate ops.  Collective.
  block_store_t(const partition_t& partition, const int rank, Array<const line_t> lines);
public:
  ~block_store_t();

  // Number of blocks
  int blocks() const {
    return block_info.size()-1;
  }

  // Estimate memory usage
  uint64_t memory_usage() const;

  // Accumulate new data into a block.  This function is thread safe.
  // If the block is complete, schedule a counting job.
  void accumulate(int local_id, RawArray<const Vector<super_t,2>> new_data);

  // Access the data for a completed block
  RawArray<const Vector<super_t,2>,4> get(section_t section, Vector<int,4> block) const;

  // Same as above, but refer to blocks via *local* block id.
  RawArray<const Vector<super_t,2>,4> get(int local_id) const;
  RawArray<const Vector<super_t,2>> get_flat(int local_id) const;

  // Count wins and losses.  Normally scheduled automatically from accumulate.
  void count_wins(int local_id);
};

// The kernel of count_wins factored out for use elsewhere
Vector<uint64_t,3> count_block_wins(const section_t section, const Vector<int,4> block, RawArray<const Vector<super_t,2>> data);

}
}
