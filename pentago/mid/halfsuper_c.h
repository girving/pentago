// Half of a super_t, by parity
#pragma once

#include "../utility/metal.h"
#include "../utility/sse.h"
#include "../utility/wasm.h"
#include "../utility/zero.h"

typedef struct halfsuper_s_ {
#if PENTAGO_SSE
  __m128i x;
#else
  uint64_t a, b;
#endif
} halfsuper_s;
