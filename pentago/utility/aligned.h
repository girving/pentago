// Aligned array allocation
//
// For performance reasons, it is useful for certain arrays to be aligned to cache
// lines even if they don't need to be for correctness.  The most important example
// of this is Vector<super_t,2>, the core data type in the endgame solver.  This
// type takes up 64 bytes, exactly the cache line size on most modern machines.
#pragma once

#include <type_traits>
#include "pentago/utility/array.h"
namespace pentago {

// Allocate an aligned buffer with the given properties, and return buffer and owner.
shared_ptr<void> aligned_buffer_helper(size_t alignment, size_t size);

// Allocate an aligned, uninitialize array of the given type and shape
template<class T,int d> Array<T,d> aligned_buffer(const Vector<int,d>& shape) {
  static_assert(std::is_trivially_destructible<T>::value,"");
  static_assert(!(sizeof(T)&(sizeof(T)-1)),"size must be a power of two");
  auto raw = aligned_buffer_helper(std::max(sizeof(T),sizeof(void*)),sizeof(T)*shape.product());
  return Array<T,d>(shape, shared_ptr<T>(raw, static_cast<T*>(raw.get())));
}

template<class T> Array<T> aligned_buffer(const int size) {
  return aligned_buffer<T>(vec(size));
}

}
