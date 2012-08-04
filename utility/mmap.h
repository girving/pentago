// Array allocation using mmap
#pragma once

#include <other/core/array/Array.h>
#include <other/core/structure/Tuple.h>
#include <boost/noncopyable.hpp>
namespace pentago {

using namespace other;

// Allocate a large buffer via mmap and return buffer and owner.
Tuple<void*,PyObject*> mmap_buffer_helper(size_t size);

// Allocate a large buffer using mmap.  The data will be zero initialized.
template<class T> static inline Array<T> mmap_buffer(int size) {
  BOOST_MPL_ASSERT((boost::has_trivial_destructor<T>));
  auto buffer = mmap_buffer_helper(sizeof(T)*size);
  Array<T> array(size,(T*)buffer.x,buffer.y);
  OTHER_XDECREF(buffer.y);
  return array;
}

// Same as above, but for Array<T,d>
template<class T,int d> static inline Array<T,d> mmap_buffer(const Vector<int,d>& shape) {
  Array<T> flat = mmap_buffer<T>(shape.product());
  return Array<T,d>(shape,flat.data(),flat.borrow_owner());
}

}
