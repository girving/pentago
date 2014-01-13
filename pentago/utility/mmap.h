// Array allocation using mmap
#pragma once

#include <geode/array/Array.h>
#include <geode/structure/Tuple.h>
#include <boost/noncopyable.hpp>
namespace pentago {

using namespace geode;

// Allocate a large buffer via mmap and return buffer and owner.
GEODE_EXPORT Tuple<void*,PyObject*> mmap_buffer_helper(size_t size);

// Allocate a large buffer using mmap.  The data will be zero initialized.
template<class T> static inline Array<T> mmap_buffer(int size) {
  static_assert(has_trivial_destructor<T>::value,"");
  auto buffer = mmap_buffer_helper(sizeof(T)*size);
  Array<T> array(size,(T*)buffer.x,buffer.y);
  GEODE_XDECREF(buffer.y);
  return array;
}

// Same as above, but for Array<T,d>
template<class T,int d> static inline Array<T,d> mmap_buffer(const Vector<int,d>& shape) {
  Array<T> flat = mmap_buffer<T>(shape.product());
  return Array<T,d>(shape,flat.data(),flat.borrow_owner());
}

}
