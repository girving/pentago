// ceil_div(a,b) = ceil(a/b), but with integers
#pragma once

#include <geode/utility/type_traits.h>
#include <cassert>
namespace pentago {

template<class TV,class T> static inline TV ceil_div(TV a, T b) {
  static_assert(is_integral<T>::value,"");
  assert(b > 0);
  return (a+b-1)/b;
}

}
