// Endian utilities
#pragma once

#ifndef __wasm__
#include <bit>
#endif
namespace pentago {

#ifdef __wasm__

#define PENTAGO_LITTLE_ENDIAN  // Emscripten is little endian

#else

// Assume little endian
#define PENTAGO_LITTLE_ENDIAN

template<class T> static inline T endian_reverse(const T& data) {
  die("Only little endian is supported");
}

template<class T> static inline const T& native_to_little_endian(const T& data) {
  static_assert(std::endian::native == std::endian::little, "Only little endian is supported");
  return data;
}

template<class T> static inline const T& little_to_native_endian(const T& data) {
  static_assert(std::endian::native == std::endian::little, "Only little endian is supported");
  return data;
}

template<class A> void to_little_endian_inplace(const A& data) {
  static_assert(std::endian::native == std::endian::little, "Only little endian is supported");
}

#endif  // __wasm__

}  // namespace pentago
