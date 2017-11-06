// Load balancing statistics (see partition.h for actual balancing)
//
// These routines are used both at runtime to monitor current load balancing
// at offline by the "predict" script to analyze load balancing before a job
// is launched.  Note that this depends critically on load balancing being
// sufficiently fast and lightweight for computation on a single rank.
#pragma once

#include "pentago/end/partition.h"
#include "pentago/end/reduction.h"
#include "pentago/utility/box.h"
namespace pentago {
namespace end {

struct load_balance_t : public boost::noncopyable {
  Box<int64_t> lines, line_blocks, line_nodes; // Compute counts
  Box<int64_t> blocks, block_nodes, block_local_ids; // Owned block counts

  load_balance_t();
  ~load_balance_t();

  Range<Box<int64_t>*> boxes();
  void enlarge(const load_balance_t& load);

  // Print load balancing statistics
  void print() const;
};

// Compute load balance information given rank_lines and rank_blocks.  The result is valid only on the root.
shared_ptr<const load_balance_t> load_balance(
    const reduction_t<int64_t,max_op>& reduce_max, RawArray<const line_t> lines,
    RawArray<const local_block_t> blocks);

// Compute local balance information in a single process environment
shared_ptr<const load_balance_t> serial_load_balance(const shared_ptr<const partition_t>& partition);

// Create an empty partition, typically for sentinel use
shared_ptr<const partition_t> empty_partition(const int ranks, const int slice);

}
}
