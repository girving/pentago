// Endgame computation structure code with overlapped communication and compute
#pragma once

#include <pentago/mpi/block_store.h>
namespace pentago {
namespace mpi {

struct line_data_t;

// Set of computed lines whose data is ready to be sent.
struct finished_lines_t : public boost::noncopyable {
  spinlock_t lock;
  // Each of these pointers is owning, so the reference count must be decremented when done.
  // This happens automatically as a consequence of line_data_t::decrement_unsent_output_blocks.
  vector<line_data_t*> lines;

  finished_lines_t();
  ~finished_lines_t();
};

// Estimate base memory usage of compute (ignoring active lines)
uint64_t base_compute_memory_usage(const int lines);

// Compute all given lines.  Each line requires a number of blocks of data from other processors,
// so for efficiency we overlap communication and compute.  The computation will usually be mostly
// sequential, but in case communication occurs in a strange order we speculatively ask for inputs
// for a number of lines in parallel, and compute whenever inputs are ready.
//
// The number of lines to speculate is controlled by an arbitrary memory limit in bytes.
void compute_lines(const MPI_Comm comm, const Ptr<const block_store_t> input_blocks, block_store_t& output_blocks, Array<const line_t> lines, const uint64_t memory_limit);

}
}
