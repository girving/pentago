// WebAssembly machinery
#pragma once

namespace pentago {

#ifdef __wasm__
#define WASM_EXPORT extern "C"
#define WASM_IMPORT extern "C"
#else
#define WASM_EXPORT
#define WASM_IMPORT
#endif

WASM_IMPORT void wasm_log(const char* name, const double value);

}  // namespace pentago
