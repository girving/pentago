// Minimal inttypes.h for the freestanding liblzma wasm build: clang's own inttypes.h wraps this
// one via #include_next.  liblzma's sysdefs.h defines the PRI* macros it needs if absent.
#pragma once
#include <stdint.h>
