// Turn a Vector<T,d> component into another array dimension
#pragma once

#include "pentago/utility/array.h"
namespace pentago {

template<class T,int d> static RawArray<T,d> scalar_view(RawArray<T,d> x) {
  return x;
}

template<class T,int m,int d> static RawArray<T,d+1> scalar_view(RawArray<Vector<T,m>,d> x) {
  return RawArray<T,d+1>(concat(x.shape(), vec(m)), x.data()->data());
}

template<class T,int m,int d> static RawArray<const T,d+1>
scalar_view(RawArray<const Vector<T,m>,d> x) {
  return RawArray<const T,d+1>(concat(x.shape(), vec(m)), x.data()->data());
}

template<class A> static auto scalar_view(const A& x) {
  return scalar_view(x.raw());
}

}  // namespace pentago
