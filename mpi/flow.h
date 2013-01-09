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
  MPI_Comm request_comm; // Requests for one of our local input blocks.  Tag = request_id, data = int dimensions, int response_tag.
  MPI_Comm response_comm; // Responses to our block requests complete with input block data.  Tag = response_tag, block data.
  MPI_Comm output_comm; // Output data to be merged into one of our local output blocks.  Tag = request_id, block data.
#if !PENTAGO_MPI_FUNNEL
  MPI_Comm wakeup_comm; // Wake up messages from a worker thread to the communication thread when a line finishes, possibly after block compression.  Tag = compress_blocks?b:0, data = &line.
#endif

  // Note: In compressed mode, response messages are compressed but output messages are uncompressed.  This is because
  // (1) the messages is stored compressed by the owner, so sending compressed is easy and (2) most output messages would need
  // to be temporarily uncompressed at the owner anyways to be combined with previous contributions.

  explicit flow_comms_t(MPI_Comm comm);
  ~flow_comms_t();
};

// Associated tags
const int barrier_tag = 1111; // No data

// Compute all given lines.  Each line requires a number of blocks of data from other processors,
// so for efficiency we overlap communication and compute.  The computation will usually be mostly
// sequential, but in case communication occurs in a strange order we speculatively ask for inputs
// for a number of lines in parallel, and compute whenever inputs are ready.
//
// The number of lines to speculate is controlled by (1) an arbitrary memory limit in bytes and
// (2) a limit on the number of lines in communication at any given time.
void compute_lines(const flow_comms_t& comms, const Ptr<const block_store_t> input_blocks, block_store_t& output_blocks, RawArray<const line_t> lines, const uint64_t memory_limit, const int line_gather_limit, const int line_limit);

}
}
