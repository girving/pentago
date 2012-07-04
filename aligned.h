// Aligned array allocation
#pragma once

#include <other/core/array/Array.h>
#include <boost/noncopyable.hpp>
namespace pentago {

using namespace other;
using std::bad_alloc;

struct aligned_buffer_t : public boost::noncopyable
{
  OTHER_DECLARE_TYPE // Declare pytype
  PyObject_HEAD // Reference counter and pointer to type object
  char* data; // Owning pointer to data

  aligned_buffer_t(char* data)
    : data(data) {
    (void)PyObject_INIT(this,&pytype);
  }
};

// Allocate an aligned, uninitialized array of the given type and size
template<class T> Array<T> aligned_buffer(int size) {
  BOOST_MPL_ASSERT((boost::has_trivial_destructor<T>));
  char* data = (char*)malloc(sizeof(T)*size);
  if (!data) throw bad_alloc();
  auto* buffer = new aligned_buffer_t(data);
  Array<T> array(size,(T*)data,(PyObject*)buffer);
  OTHER_XDECREF(buffer);
  return array;
}

// Allocate an aligned, uninitialize array of the given type and shape
template<class T,int d> Array<T,d> aligned_buffer(const Vector<int,d>& shape) {
  Array<T> flat = aligned_buffer<T>(shape.product());
  return Array<T,d>(shape,flat.data(),flat.borrow_owner());
}

}
