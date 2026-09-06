// Minimal assert.h for the freestanding liblzma wasm build: we compile with NDEBUG
#pragma once
#define assert(condition) ((void)0)
