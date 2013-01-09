// Core compute kernels for the MPI code
#pragma once

#include <pentago/mpi/line.h>
#include <pentago/mpi/config.h>
#include <pentago/symmetry.h>
#include <pentago/utility/counter.h>
#include <pentago/utility/spinlock.h>
#include <other/core/array/Array4d.h>
#include <boost/noncopyable.hpp>
namespace pentago {
namespace mpi {

/* Notes:
 *
 * 1. Unlike the out-of-core solver, here we store both input and output data in block major order (essentially a 5D array).
 *    Furthermore, the compute kernel consumes data from the input array in native format, accounting for axis permutation
 *    and symmetries on the fly.  As a consequence, both input and output blocks are available as flat slices of input or
 *    output without copying or rearrangement, avoiding extra buffering.
 *
 * 2. Each line starts out unallocated.  Eventually, four things happen:
 *
 *    Allocation: Sufficient memory exists, so input and output buffers are allocated.  Requests are sent out to whoever owns the input blocks.
 *    Scheduling: All input blocks arrive, so all microlines are scheduled for execution on worker threads.
 *    Completion: All microlines complete.  A wakeup request is sent to the communication thread, which posts sends for each output block.
 *    Deallocation: All output sends complete, so the line is deallocated.
 */

struct line_data_t {
  // Line and child line data
  const line_t& line;
  const Vector<int,4> input_shape; // Unstandardized
  const Vector<int,4> output_shape;
  const uint64_t memory_usage;

  line_data_t(const line_t& line);
  ~line_data_t();
};

struct line_details_t : public boost::noncopyable {
  // Initial information
  const line_data_t pre;
  const event_t line_event;
  const int block_stride; // Number of nodes in all blocks except possible the last

  // Standardization
  const section_t standard_child_section;
  const uint8_t section_transform; // Maps output space to input (child) space
  const Vector<int,4> permutation; // Child quadrant i maps to quadrant permutation[i]
  const unsigned child_dimension : 2;
  const uint8_t input_blocks;
  const Vector<uint8_t,4> first_child_block; // First child block
  const symmetry_t inverse_transform;

  // Rotation minimal quadrants
  const Vector<RawArray<const quadrant_t>,4> rmin;

  // Symmetries needed to restore minimality after reflection
  const Array<const local_symmetry_t> all_reflection_symmetries;
  const Vector<int,5> reflection_symmetry_offsets;

  // Number of blocks we need before input is ready, microlines left to compute, and output blocks left to send.
  int missing_input_responses;
  counter_t missing_input_blocks;
  counter_t missing_microlines;
  counter_t unsent_output_blocks;

  // Information needed to account for reflections in input block data
  const Vector<int,4> reflection_moves;

  // Input and output data.  Both are stored in 5D order where the first dimension
  // is the block, to avoid copying before and after compute.
  const Array<Vector<super_t,2>> input, output;

  // When computation is complete, send a wakeup message here
#if PENTAGO_MPI_FUNNEL
  typedef BOOST_PP_IF(PENTAGO_MPI_COMPRESS_OUTPUTS,int,unit) wakeup_block_t;
  typedef function<void(line_details_t*,wakeup_block_t)> wakeup_t;
  const wakeup_t wakeup;
#else
  const MPI_Comm wakeup_comm;
  const line_details_t* const self; // Buffer for wakeup message
#endif

  line_details_t(const line_data_t& pre, BOOST_PP_IF(PENTAGO_MPI_FUNNEL,const wakeup_t& wakeup,const MPI_Comm wakeup_comm));
  ~line_details_t();

  // Get the kth input block
  Vector<uint8_t,4> input_block(int k) const;

  // Extract the data for the kth block of either the input or output array.
  // In compressed mode, input block data has an extra entry to account for possible expansion.
  RawArray<Vector<super_t,2>> input_block_data(int k) const;
  RawArray<Vector<super_t,2>> input_block_data(Vector<uint8_t,4> block) const;
#if PENTAGO_MPI_COMPRESS_OUTPUTS
  RawArray<const uint8_t> compressed_output_block_data(int k) const;
private:
  friend void compress_output_block(line_details_t* const line, const int b);
  RawArray<Vector<super_t,2>> output_block_data(int k) const;
public:
#else
  RawArray<Vector<super_t,2>> output_block_data(int k) const;
#endif

  // Call this whenever a new input response arrives, but possibly *before* the data has been moved into place.
  // Returns the number of messages remaining.  Used by flow.cpp to throttle the number of simultaneous line gathers.
  // Warning: *Not* thread safe, so call only from flow.cpp.
  int decrement_input_responses();

  // Call this whenever a new block is in place.  Used as a request callback for when MPI_Irecv's finish.
  // When the count hits zero, the line will be automatically schedule.
  void decrement_missing_input_blocks();

  // Decrement the number of unsent output blocks, and return the number remaining.
  int decrement_unsent_output_blocks();
};

// Schedule a line computation (called once all input blocks are in place)
void schedule_compute_line(line_details_t& line);

}
}
