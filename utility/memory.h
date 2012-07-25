// Memory usage estimation
#pragma once

#include <other/core/array/forward.h>
#include <other/core/python/forward.h>
#include <other/core/structure/forward.h>
namespace pentago {

template<class T> static inline uint64_t memory_usage(const T& object) {
  return object.memory_usage();
}

template<class T> static inline uint64_t memory_usage(const Ref<T>& object) {
  return memory_usage(*object);
}

template<class T> static inline uint64_t memory_usage(const Ptr<T>& object) {
  return object?memory_usage(*object):0;
}

template<class T> static inline uint64_t memory_usage(const Array<T>& array) {
  return sizeof(T)*array.size();
}

template<class TK,class T> static inline uint64_t memory_usage(const Hashtable<TK,T>& table) {
  return sizeof(HashtableEntry<TK,T>)*table.max_size();
}

}