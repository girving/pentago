// Aligned array allocation
//
// For performance reasons, it is useful for certain arrays to be aligned to cache
// lines even if they don't need to be for correctness.  The most important example
// of this is Vector<super_t,2>, the core data type in the endgame solver.  This
// type takes up 64 bytes, exactly the cache line size on most modern machines.
#pragma once

#include <geode/array/Array.h>
#include <geode/structure/Tuple.h>
namespace pentago {

using namespace geode;

// Allocate an aligned buffer with the given properties, and return buffer and owner.
GEODE_EXPORT Tuple<void*,PyObject*> aligned_buffer_helper(size_t alignment, size_t size);

// Allocate an aligned, uninitialized array of the given type and size
template<class T> Array<T> aligned_buffer(int size) {
  static_assert(has_trivial_destructor<T>::value,"");
  static_assert(!(sizeof(T)&(sizeof(T)-1)),"size must be a power of two");
  auto aligned = aligned_buffer_helper(max(sizeof(T),sizeof(void*)),sizeof(T)*size);
  Array<T> array(size,(T*)aligned.x,aligned.y);
  GEODE_XDECREF(aligned.y);
  return array;
}

// Allocate an aligned, uninitialize array of the given type and shape
template<class T,int d> Array<T,d> aligned_buffer(const Vector<int,d>& shape) {
  Array<T> flat = aligned_buffer<T>(shape.product());
  return Array<T,d>(shape,flat.data(),flat.borrow_owner());
}

}
