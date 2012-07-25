// Endgame computation structure code with overlapped communication and compute
#pragma once

#include <pentago/mpi/block_store.h>
namespace pentago {
namespace mpi {

struct line_data_t;

// Estimate base memory usage of compute (ignoring active lines)
uint64_t base_compute_memory_usage(const int lines);

// The various communications used by compute_lines, and their associated messages
struct flow_comms_t : public boost::noncopyable {
  const int rank;
  MPI_Comm barrier_comm; // Barrier synchronization messages passed down to ibarrier_t.  Tag = barrier_tag, no data.
  MPI_Comm request_comm; // Requests for one of our local input blocks.  Tag = owner block id, no data.
  MPI_Comm response_comm; // Responses to our block requests complete with input block data.  Tag = owner block id, block data.
  MPI_Comm output_comm; // Output data to be merged into one of our local output blocks.  Tag = owner block id, block data.
  MPI_Comm wakeup_comm; // Wake up messages from a worker thread to the communication thread when a line finishes.  Tag = wakeup_tag, data = &line

  explicit flow_comms_t(MPI_Comm comm);
  ~flow_comms_t();
};

// Associated tags
const int barrier_tag = 1111; // No data
const int wakeup_tag = 2222; // data = &line, type = MPI_LONG_LONG_INT

// Compute all given lines.  Each line requires a number of blocks of data from other processors,
// so for efficiency we overlap communication and compute.  The computation will usually be mostly
// sequential, but in case communication occurs in a strange order we speculatively ask for inputs
// for a number of lines in parallel, and compute whenever inputs are ready.
//
// The number of lines to speculate is controlled by an arbitrary memory limit in bytes.
void compute_lines(const flow_comms_t& comms, const Ptr<const block_store_t> input_blocks, block_store_t& output_blocks, RawArray<const line_t> lines, const uint64_t memory_limit);

}
}
