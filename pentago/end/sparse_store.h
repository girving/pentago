// Sparse storage of a bunch of arrays taking advantage of virtual memory
#pragma once

/* I'm afraid of malloc fragmentation.  Storing compressed blocks involves
 * a large number of randomly sized arrays, the total size of which will
 * push up against the total memory in the machine.  Therefore, we use the
 * following trick I got from Eftychis Sifakis:
 *
 * 1. Allocate way more memory than you need via a single, gigantic mmap.
 * 2. Never touch the parts you don't need.
 *
 * As a consequence, we get to use all flat buffers, and everything is simple.
 * The main downside is that our memory usage per block will be the maximum
 * of the size of the compressed partially accumulated arrays.
 *
 * This class is thread safe as long as two threads don't try to write one
 * array at the same time.
 */

#include <pentago/base/superscore.h>
#include <pentago/utility/thread.h>
#include <geode/array/Array.h>
namespace pentago {
namespace end {

class sparse_store_t : public Noncopyable {
  struct sizes_t {
    int size; // Current size of the array
    int peak_size; // Peak size
  };
  const Array<sizes_t> sizes; // Sizes of each array

  const int max_array_size; // Rounded up to the nearest page
  const size_t sparse_size; // count*max_array_size
  uint8_t* const sparse_base; // mmap'ed memory region, only part of which will ever be resident in memory
public:

  // Allocates count empty arrays.  No array can grow beyond the given limit.
  sparse_store_t(const int count, const int max_array_size);
  ~sparse_store_t();

  // Current memory usage
  uint64_t current_memory_usage() const;

  // Estimate memory usage based on a guess at the total size
  static uint64_t estimate_peak_memory_usage(int count, uint64_t total_size);

  int size(int array) const {
    return sizes[array].size;
  }

  int peak_size(int array) const {
    return sizes[array].peak_size;
  }

  void set_size(int array, int size) const;

  // Get access to the entire buffer available to this array.  This must be used with care:
  // most of it is likely to be non-resident, so looping through the entire thing could cause
  // massive swapping.  Write or read only the part you need, then call set_size if necessary.
  RawArray<uint8_t> whole_buffer(int array) const;

  // Get access to the correctly sized, currently allocated buffer.
  RawArray<uint8_t> current_buffer(int array) const;

  // Replace the given array with the compressed version of the given buffer.  The input array is destroyed.
  void compress_and_set(int array, RawArray<Vector<super_t,2>> uncompressed, event_t event);
};

}
}
