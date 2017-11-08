// Core compute kernels for the MPI code
//
// Since the endgame solver operates near the limit of available RAM, it is
// critical to minimize the amount of working memory required during computation.
// To do this, moves in different quadrants are analyzed on potentially different
// machines, and combined together after communication.  The basic unit of work
// becomes a "block line", a set of positions which share the same block in all
// dimensions but one.  By considering only those moves that change the uncommon
// quadrant, the set of inputs for a given block line is exactly another block line.
#pragma once

#include "pentago/end/config.h"
#include "pentago/end/line.h"
#include "pentago/base/symmetry.h"
#include "pentago/utility/counter.h"
#include "pentago/utility/spinlock.h"
#include "pentago/utility/array.h"
#include "pentago/utility/unit.h"
namespace pentago {
namespace end {

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
  typedef std::conditional_t<PENTAGO_MPI_COMPRESS_OUTPUTS,int,unit_t> wakeup_block_t;
  typedef function<void(line_details_t&,wakeup_block_t)> wakeup_t;
  const wakeup_t wakeup;
  const line_details_t* const self; // Buffer for wakeup message

  line_details_t(const line_data_t& pre, const wakeup_t& wakeup);
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
