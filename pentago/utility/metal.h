// Metal utilities
#pragma once

#include "pentago/utility/wasm.h"
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdbool.h>
#include <stdint.h>
#include <assert.h>
#ifdef __cplusplus
#include <algorithm>
#endif
#endif
NAMESPACE_PENTAGO

// Are we in normal C++?
#if defined(__cplusplus) && !defined(__METAL_VERSION__)
#define PENTAGO_CPP 1
#else
#define PENTAGO_CPP 0
#endif

// Metal keywords, if we need them
#ifdef __METAL_VERSION__
#define METAL_CONSTANT constant
#define METAL_DEVICE device
#define METAL_GLOBAL constant
#define METAL_INLINE
#define NONMETAL_ASSERT(...) ((void)0)
#else
#define METAL_CONSTANT
#define METAL_DEVICE
#define METAL_GLOBAL const
#define METAL_INLINE static inline
#define NONMETAL_ASSERT(...) assert(__VA_ARGS__)
#endif

// Functions in metal:: or std::
#ifdef __cplusplus
#ifdef __METAL_VERSION__

using metal::min;
using metal::popcount;
template<class T> struct is_unsigned;
template<class T> struct make_signed;
template<class T> struct make_unsigned;
#define SIGNS(I, UI) \
  template<> struct is_unsigned<UI> { const bool value = true; }; \
  template<> struct make_signed<UI> { typedef I type; }; \
  template<> struct make_signed<I> { typedef I type; }; \
  template<> struct make_unsigned<I> { typedef UI type; }; \
  template<> struct make_unsigned<UI> { typedef UI type; };
SIGNS(int16_t, uint16_t)
SIGNS(int32_t, uint32_t)
SIGNS(int64_t, uint64_t)
#undef SIGNS

#else  // if non-metal C++

using std::min;
using std::is_unsigned;
using std::make_signed;
using std::make_unsigned;

#endif
#endif

END_NAMESPACE_PENTAGO
