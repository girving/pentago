// Array ranges
#pragma once

#include "pentago/utility/array.h"
namespace pentago {

template<class I> static inline Array<I> arange(const I n) {
  static_assert(std::is_integral<I>::value);
  GEODE_ASSERT(n >= 0);
  Array<I> result(n, uninit);
  for (const auto i : range(n)) {
    result[i] = i;
  }
  return result;
}

}
