// Conversion between flat and multidimensional indices
#pragma once

#include "pentago/utility/vector.h"
namespace pentago {

// Flatten a multidimensional index
template<int d> static inline int index(const Vector<int,d>& shape, const Vector<int,d>& I) {
  if (d == 0) return 0;
  GEODE_DEBUG_ONLY(for (int i = 0; i < d; i++) assert(I[i] < shape[i]);)
  int index = I[0];
  for (int i = 1; i < d; i++)
    index = index * shape[i] + I[i];
  return index;
}

// Unpack a flat index into its multidimensional source
template<int d> static inline Vector<int,d> decompose(const Vector<int,d>& shape, int I) {
  Vector<int,d> r;
  for (int i = d-1; i > 0; i--) { 
    const int s = shape[i];
    const int j = I / s;
    r[i] = I - j * s;
    I = j;
  }
  r[0] = I;
  return r;
}

// Compute strides for a flat multidimensional array
template<int d> static inline Vector<int,d> strides(const Vector<int,d>& shape) {
  Vector<int,d> strides;
  if (d) strides[d-1] = 1;
  for (int i = d-2; i >= 0; i--)
    strides[i] = shape[i+1] * strides[i+1];
  return strides;
}

}
