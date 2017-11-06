// popcount
#pragma once

#include <cstdint>
namespace pentago {

static inline int popcount(uint16_t n) {
  return __builtin_popcount(n);
}

static inline int popcount(uint32_t n) {
  static_assert(sizeof(int)==4,"");
  return __builtin_popcount(n);
}

static inline int popcount(uint64_t n) {
#if __SIZEOF_LONG__ == 8
  static_assert(sizeof(long)==8,"");
  return __builtin_popcountl(n);
#elif __SIZEOF_LONG_LONG__ == 8
  static_assert(sizeof(long long)==8,"");
  return __builtin_popcountll(n);
#else
  #error "Can't deduce __builtin_popcount for uint64_t"
#endif
}

}
