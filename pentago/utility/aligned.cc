// Aligned array allocation

#include "pentago/utility/aligned.h"
#include "pentago/utility/debug.h"
#include "pentago/utility/memory.h"
namespace pentago {

using std::vector;
using std::bad_alloc;

shared_ptr<void> aligned_buffer_helper(size_t alignment, size_t size) {
  if (!size) return nullptr;
#ifndef __APPLE__
#ifdef __bgq__
  // See https://wiki.alcf.anl.gov/parts/index.php/Blue_Gene/Q#Allocating_Memory
  alignment = max(alignment,size_t(32));
#endif
  void* start;
  if (posix_memalign(&start,alignment,size)) THROW(bad_alloc);
  void* pointer = start;
#else
  // Mac OS 10.7.4 has a buggy version of posix_memalign, so do our own alignment
  // at the cost of one extra element
  void* start = malloc(size+alignment-1);
  if (!start) THROW(bad_alloc);
  size_t p = (size_t)start;
  p = (p+alignment-1)&~(alignment-1);
  void* pointer = (void*)p;
#endif
  report_large_alloc(size);
  return shared_ptr<void>(pointer, [size, start](void*) {
    free(start);
    report_large_alloc(-size);
  });
}

}
