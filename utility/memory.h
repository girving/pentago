// Memory usage estimation and reporting
#pragma once

#include <other/core/array/forward.h>
#include <other/core/python/forward.h>
#include <other/core/structure/forward.h>
#include <string>
namespace other {
template<class TK,class T> struct HashtableEntry;
}
namespace pentago {

using namespace other;
using std::string;

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

template<class T,int d> static inline uint64_t memory_usage(const Array<T,d>& array) {
  return sizeof(T)*array.flat.size();
}

template<class TK,class T> static inline uint64_t memory_usage(const Hashtable<TK,T>& table) {
  return sizeof(HashtableEntry<TK,T>)*table.max_size();
}

// Generate a process memory usage report
string memory_report();

// Note a large allocation or deallocation
void report_large_alloc(ssize_t change);


}
