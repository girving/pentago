// Memory usage estimation, allocation, and reporting
#pragma once

#include <pentago/utility/aligned.h>
#include <geode/array/forward.h>
#include <geode/python/forward.h>
#include <geode/structure/forward.h>
#include <string>
namespace pentago {

using namespace geode;
using std::string;

// Estimate memory usage of various objects

template<class T> static inline uint64_t memory_usage(const T& object) {
  return object.memory_usage();
}

template<class T> static inline uint64_t memory_usage(const Ref<T>& object) {
  return memory_usage(*object);
}

template<class T> static inline uint64_t memory_usage(const Ptr<T>& object) {
  return object?memory_usage(*object):0;
}

// Arguably, RawArrays don't take up any of their own memory, but that version of the function would be useless.
template<class T> static inline uint64_t memory_usage(const RawArray<T>& array) {
  return sizeof(T)*array.size();
}

template<class T> static inline uint64_t memory_usage(const Array<T>& array) {
  return sizeof(T)*array.size();
}

template<class T,int d> static inline uint64_t memory_usage(const Array<T,d>& array) {
  return sizeof(T)*array.flat.size();
}

template<class TK,class T> static inline uint64_t memory_usage(const Hashtable<TK,T>& table) {
  return sizeof(HashtableEntry<TK,T>)*table.max_size();
}

// Allocate a large array.  For now, based on profiling, we use aligned_buffer.  For a while I
// though mmap was dramatically improving fragmentation performance, but it turns out there
// were other bugs causing massive extra allocation.  If we later decide that mmap is indeed
// better, it'll be easy to flip the template alias.
template<class T> static inline Array<T> large_buffer(int size, bool zero) {
  Array<T> buffer = aligned_buffer<T>(size);
  if (zero)
    memset(buffer.data(),0,sizeof(T)*size);
  return buffer;
}

// Extract memory usage information
GEODE_EXPORT Array<uint64_t> memory_info();

// Generate a process memory usage report
GEODE_EXPORT string memory_report(RawArray<const uint64_t> info);

// Note a large allocation or deallocation
GEODE_EXPORT void report_large_alloc(ssize_t change);


}
