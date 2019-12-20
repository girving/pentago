// Memory allocation and reporting
#pragma once

#include "pentago/utility/aligned.h"
namespace pentago {

// Allocate a large array.  For now, based on profiling, we use aligned_buffer.  For a while I
// though mmap was dramatically improving fragmentation performance, but it turns out there
// were other bugs causing massive extra allocation.  If we later decide that mmap is indeed
// better, it'll be easy to flip the template alias.
template<class T> static inline Array<T> large_buffer(const int size) {
  Array<T> buffer = aligned_buffer<T>(size);
  memset(buffer.data(),0,sizeof(T)*size);
  return buffer;
}
template<class T> static inline Array<T> large_buffer(const int size, Uninit) {
  return aligned_buffer<T>(size);
}

// Extract memory usage information
Array<uint64_t> memory_info();

// Generate a process memory usage report
string memory_report(RawArray<const uint64_t> info);

// Note a large allocation or deallocation
void report_large_alloc(ssize_t change);

}
