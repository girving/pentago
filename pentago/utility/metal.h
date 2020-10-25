// Metal utilities
#pragma once

#include "pentago/utility/wasm.h"
#ifndef __METAL_VERSION__
#include <stdbool.h>
#include <stdint.h>
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
#define METAL_GLOBAL constant
#else  // !__METAL_VERSION__
#define METAL_CONSTANT
#define METAL_GLOBAL const
#endif

END_NAMESPACE_PENTAGO
