// Fast subset counting, indexing, and generation
#pragma once

#include "../utility/wasm.h"

// A k-subsets of [0,n-1], packed into 64-bit ints with 5 bits for each entry.
typedef uint64_t set_t;

typedef struct sets_t_ {
  int n, k;
  int size;
} sets_t;

// List empty spots as bit indices into side_t
typedef struct empty_t_ {
  uint8_t empty[18];
} empty_t;
