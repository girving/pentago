// WebAssembly allocation
#pragma once

#include "array.h"
#include "wasm.h"
#include <cstddef>
#if defined(__wasm__) && !defined(__APPLE__)
namespace pentago {

WASM_EXPORT __attribute__((cold)) void* malloc(size_t size);

static inline void free(void* p) {}

// Everything allocated after a mark can be discarded at once by rewinding to it
size_t wasm_heap_mark();
void wasm_heap_rewind(const size_t mark);

template<class T> RawArray<T> wasm_buffer(const int size) {
  return RawArray<T>(size, reinterpret_cast<T*>(malloc(sizeof(T) * size)));
}

}  // namespace pentago
#endif  // defined(__wasm__) && !defined(__APPLE__)
