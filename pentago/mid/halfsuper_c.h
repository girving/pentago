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

typedef struct halfsupers_t_ {
  halfsuper_s win, notlose;
} halfsupers_t;

_Static_assert(sizeof(halfsuper_s) == 16, "");
_Static_assert(sizeof(halfsupers_t) == 32, "");
