// WebAssembly allocation
#pragma once

#include "pentago/utility/array.h"
#include "pentago/utility/wasm.h"
#include <cstddef>
#ifdef __wasm__
namespace pentago {

WASM_EXPORT void* malloc(size_t size);

template<class T> RawArray<T> wasm_buffer(const int size) {
  return RawArray<T>(size, reinterpret_cast<T*>(malloc(sizeof(T) * size)));
}

}  // namespace pentago
#endif  // __wasm__
