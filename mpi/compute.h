// Core compute kernels for the MPI code
#pragma once

#include <pentago/mpi/line.h>
#include <pentago/mpi/config.h>
#include <pentago/symmetry.h>
#include <pentago/utility/spinlock.h>
#include <other/core/array/Array4d.h>
#include <boost/noncopyable.hpp>
#include <boost/scoped_ptr.hpp>
namespace pentago {
namespace mpi {

using namespace other;
using boost::scoped_ptr;
struct allocated_t;

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

struct line_data_t : public boost::noncopyable {
  // Line and child line data
  const line_t& line;
  const Vector<int,4> input_shape; // Unstandardized
  const Vector<int,4> output_shape;

  // Valid only after allocate is called
  scoped_ptr<allocated_t> rest;

  line_data_t(const line_t& line);
  ~line_data_t();

  // Valid before allocate is called
  uint64_t memory_usage() const;

  // Prepare to compute.
  void allocate(MPI_Comm wakeup_comm);

  // Information about the child
  section_t standard_child_section() const;

  // Get the kth input block
  int input_blocks() const;
  Vector<int,4> input_block(int k) const;

  // Extract the data for the kth block of either the input or output array
  RawArray<Vector<super_t,2>> input_block_data(int k) const;
  RawArray<Vector<super_t,2>> input_block_data(Vector<int,4> block) const;
  RawArray<const Vector<super_t,2>> output_block_data(int k) const;

  // Call this whenever a new block is in place.  Used as a request callback for when MPI_Irecv's finish.
  // When the count hits zero, the line will be automatically schedule.
  void decrement_missing_input_blocks();

  // Decrement the number of unsent output blocks, and return the number remaining.
  int decrement_unsent_output_blocks();
};

// Schedule a line computation (called once all input blocks are in place)
void schedule_compute_line(line_data_t& line);

}
}
