// Memory usage estimation
#pragma once

#include "pentago/utility/array.h"
#include <unordered_map>
#include <string>
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

}
