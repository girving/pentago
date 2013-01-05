// Load balancing statistics (see partition.h for actual balancing)
#pragma once

#include <pentago/mpi/partition.h>
#include <other/core/geometry/Box.h>
#include <other/core/utility/str.h>
namespace pentago {
namespace mpi {

using other::str;

struct load_balance_t : public Object {
  OTHER_DECLARE_TYPE(OTHER_NO_EXPORT)

  Box<int64_t> lines, line_blocks, line_nodes; // Compute counts
  Box<int64_t> blocks, block_nodes, block_local_ids; // Owned block counts

protected:
  load_balance_t();
public:
  ~load_balance_t();

  Range<Box<int64_t>*> boxes();
  void enlarge(const load_balance_t& load);
};

// Print load balancing statistics
string str(const load_balance_t& load);

// Compute load balance information given rank_lines and rank_blocks.  The result is valid only on the root.
Ref<const load_balance_t> load_balance(const MPI_Comm comm, RawArray<const line_t> lines, RawArray<const local_block_t> blocks);

// Compute local balance information in a single process environment
Ref<const load_balance_t> serial_load_balance(const partition_t& partition);

// Create an empty partition, typically for sentinel use
Ref<const partition_t> empty_partition(const int ranks, const int slice);

// Test self consistency of a partition
void partition_test(const partition_t& partition);

}
}
