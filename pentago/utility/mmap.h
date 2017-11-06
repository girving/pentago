// Array allocation using mmap
//
// Similar to aligned_buffer in aligned.h, but for particularly huge arrays.
#pragma once

#include "pentago/utility/array.h"
#include <memory>
namespace pentago {

using std::shared_ptr;

// Allocate a large buffer via mmap
shared_ptr<void> mmap_buffer_helper(size_t size);

// Allocate a large buffer using mmap.  The data will be zero initialized.
template<class T,int d> static inline Array<T,d> mmap_buffer(const Vector<int,d>& shape) {
  const auto raw = mmap_buffer_helper(sizeof(T) * shape.product());
  return Array<T,d>(shape, shared_ptr<T[]>(raw, static_cast<T*>(raw.get())));
}

Array<const uint8_t> mmap_file(const string& path);

}
