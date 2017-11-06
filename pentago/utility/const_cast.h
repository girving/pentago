#pragma once

namespace pentago {

template<class T> static inline T& const_cast_(const T& x) {
  return const_cast<T&>(x);
}

}
