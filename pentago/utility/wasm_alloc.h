// WebAssembly allocation
#pragma once

#include "pentago/utility/array.h"
#include "pentago/utility/wasm.h"
#include <cstddef>
#if defined(__wasm__) && !defined(__APPLE__)
namespace pentago {

WASM_EXPORT __attribute__((cold)) void* malloc(size_t size);

static inline void free(void* p) {}

template<class T> RawArray<T> wasm_buffer(const int size) {
  return RawArray<T>(size, reinterpret_cast<T*>(malloc(sizeof(T) * size)));
}

}  // namespace pentago
#endif  // defined(__wasm__) && !defined(__APPLE__)
