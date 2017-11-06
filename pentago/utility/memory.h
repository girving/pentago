// Memory usage estimation, allocation, and reporting
#pragma once

#include "pentago/utility/aligned.h"
#include "pentago/utility/array.h"
#include <unordered_map>
#include <string>
#include <memory>
namespace pentago {

using std::string;
using std::shared_ptr;
using std::unordered_map;

// Estimate memory usage of various objects

template<class T> static inline uint64_t memory_usage(const T& object) {
  return object.memory_usage();
}

template<class T> static inline uint64_t memory_usage(const shared_ptr<T>& object) {
  return object ? memory_usage(*object) : 0;
}

// Arguably, RawArrays don't take up any of their own memory, but that version
// of the function would be useless.
template<class T,int d> static inline uint64_t memory_usage(const RawArray<T,d>& array) {
  return sizeof(T)*array.total_size();
}

template<class T,int d> static inline uint64_t memory_usage(const Array<T,d>& array) {
  return sizeof(T)*array.total_size();
}

// Estimate from https://stackoverflow.com/questions/22498768.  Very approximate.
template<class K,class V,class H> static inline uint64_t memory_usage(const unordered_map<K,V,H>& map) {
  return (sizeof(K)+sizeof(V)+sizeof(void*))*map.size() + sizeof(void*)*map.bucket_count();
}

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
