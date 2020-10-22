// Base 2 logarithms of integers and related functions.  Inexact versions round down.
#pragma once

#include <cassert>
#include <cstdint>
namespace pentago {

template<class I> static inline int integer_log(const I n) {
  static_assert(sizeof(I) == 4 || sizeof(I) == 8);
  assert(n >= 0);
  std::make_unsigned_t<I> v = n;
  int c = 0;
  if (sizeof(v) == 8 && v & 0xffffffff00000000) { v >>= 32; c |= 32; }
  if (v & 0xffff0000) { v >>= 16; c |= 16; }
  if (v & 0xff00)     { v >>= 8;  c |= 8;  }
  if (v & 0xf0)       { v >>= 4;  c |= 4;  }
  if (v & 0xc)        { v >>= 2;  c |= 2;  }
  if (v & 2)                      c |= 1;
  return c;
}

template<class UI> static inline UI min_bit(const UI v) {
  static_assert(std::is_unsigned<UI>::value);
  typedef std::make_signed_t<UI> SI;
  return v & (UI)-(SI)v;
}

template<class UI> static inline int integer_log_exact(const UI x) {
  constexpr int n = sizeof(UI);
  static_assert(std::is_unsigned<UI>::value);
  static_assert(n == sizeof(int) || n == sizeof(long) || n == sizeof(long long));
  const int b = n == sizeof(int)  ? __builtin_ffs(x) - 1
              : n == sizeof(long) ? __builtin_ffsl(x) - 1
                                  : __builtin_ffsll(x) - 1;
  assert(x == (unsigned int)1 << b);
  return b;
}

}
