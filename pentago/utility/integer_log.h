// Base 2 logarithms of integers and related functions.  Inexact versions round down.
#pragma once

#include "pentago/utility/metal.h"
#include "pentago/utility/wasm.h"
NAMESPACE_PENTAGO

#if PENTAGO_CPP
template<class I> static inline int integer_log(const I n) {
  static_assert(sizeof(I) == 4 || sizeof(I) == 8);
  NONMETAL_ASSERT(n >= 0);
  typename make_unsigned<I>::type v = n;
  int c = 0;
  if (sizeof(v) == 8 && v & 0xffffffff00000000) { v >>= 32; c |= 32; }
  if (v & 0xffff0000) { v >>= 16; c |= 16; }
  if (v & 0xff00)     { v >>= 8;  c |= 8;  }
  if (v & 0xf0)       { v >>= 4;  c |= 4;  }
  if (v & 0xc)        { v >>= 2;  c |= 2;  }
  if (v & 2)                      c |= 1;
  return c;
}
#endif  // PENTAGO_CPP

template<class UI> METAL_INLINE UI min_bit(const UI v) {
  static_assert(is_unsigned<UI>().value, "");
  typedef typename make_signed<UI>::type SI;
  return v & (UI)-(SI)v;
}

template<class UI> METAL_INLINE int integer_log_exact(const UI x) {
  static_assert(is_unsigned<UI>().value, "");
#ifdef __METAL_VERSION__
  return popcount(x - 1);
#else
  constexpr int n = sizeof(UI);
  static_assert(n == sizeof(int) || n == sizeof(long) || n == sizeof(long long));
  const int b = n == sizeof(int)  ? __builtin_ffs(x) - 1
              : n == sizeof(long) ? __builtin_ffsl(x) - 1
                                  : __builtin_ffsll(x) - 1;
  assert(x == UI(1) << b);
  return b;
#endif
}

END_NAMESPACE_PENTAGO
