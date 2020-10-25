// Half of a super_t, by parity
#pragma once

#include "pentago/utility/metal.h"
#include "pentago/utility/sse.h"
#include "pentago/utility/wasm.h"
#include "pentago/utility/zero.h"

typedef struct halfsuper_s_ {
#if PENTAGO_SSE
  __m128i x;
#else
  uint64_t a, b;
#endif
} halfsuper_s;
