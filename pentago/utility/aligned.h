// Aligned array allocation
#pragma once

#include <geode/array/Array.h>
#include <geode/structure/Tuple.h>
#include <boost/noncopyable.hpp>
namespace pentago {

using namespace geode;

// Allocate an aligned buffer with the given properties, and return buffer and owner.
GEODE_EXPORT Tuple<void*,PyObject*> aligned_buffer_helper(size_t alignment, size_t size);

// Allocate an aligned, uninitialized array of the given type and size
template<class T> Array<T> aligned_buffer(int size) {
  BOOST_MPL_ASSERT((boost::has_trivial_destructor<T>));
  BOOST_STATIC_ASSERT(!(sizeof(T)&(sizeof(T)-1))); // size must be a power of two
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
