// WebAssembly machinery
#pragma once

#ifdef __cplusplus
namespace pentago {

#ifdef __wasm__
#define WASM_EXPORT extern "C"
#define WASM_IMPORT extern "C"
#define NAMESPACE_PENTAGO using namespace pentago;
#define END_NAMESPACE_PENTAGO
#define PENTAGO_NAMESPACE
#else
#define WASM_EXPORT
#define WASM_IMPORT
#define NAMESPACE_PENTAGO namespace pentago {
#define END_NAMESPACE_PENTAGO }
#define PENTAGO_NAMESPACE pentago
#endif

}  // namespace pentago
#else  // !__cplusplus
#define NAMESPACE_PENTAGO
#define END_NAMESPACE_PENTAGO
#endif
