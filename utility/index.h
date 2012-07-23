// Conversion between flat and multidimensional indices
#pragma once

#include <other/core/vector/Vector.h>
namespace pentago {

// Flatten a multidimensional index

static inline int index(const Vector<int,2>& shape, const Vector<int,2>& I) {
  return I.x*shape.y+I.y;
}

static inline int index(const Vector<int,3>& shape, const Vector<int,3>& I) {
  return (I.x*shape.y+I.y)*shape.z+I.z;
}

static inline int index(const Vector<int,4>& shape, const Vector<int,4>& I) {
  return ((I.x*shape.y+I.y)*shape.z+I.z)*shape.w+I.w;
}

// Unpack a flat index into its multidimensional source

static inline Vector<int,2> decompose(const Vector<int,2>& shape, const int I) {
  const int i0 = I/shape.y,
            i1 = I-i0*shape.y;
  return vec(i0,i1);
}

static inline Vector<int,3> decompose(const Vector<int,3>& shape, const int I) {
  const int i01 = I/shape.z,
            i0 = i01/shape.y,
            i1 = i01-i0*shape.y, 
            i2 = I-i01*shape.z;
  return vec(i0,i1,i2);
}

static inline Vector<int,4> decompose(const Vector<int,4>& shape, const int I) {
  const int i012 = I/shape.w,
            i01 = i012/shape.z,
            i0 = i01/shape.y,
            i1 = i01-i0*shape.y, 
            i2 = i012-i01*shape.z,
            i3 = I-i012*shape.w;
  return vec(i0,i1,i2,i3);
}

// Compute strides for a flat multidimensional array

static inline Vector<int,2> strides(const Vector<int,2>& shape) {
  return vec(shape.y,1);
}

static inline Vector<int,3> strides(const Vector<int,3>& shape) {
  return vec(shape.z*shape.y,shape.z,1);
}

static inline Vector<int,4> strides(const Vector<int,4>& shape) {
  return vec(shape.w*shape.z*shape.y,shape.w*shape.z,shape.w,1);
}

}
