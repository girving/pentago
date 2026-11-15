// Fast subset counting, indexing, and generation
#pragma once

#include "../utility/metal.h"
#include "../utility/wasm.h"

// A k-subsets of [0,n-1], packed into 64-bit ints with 5 bits for each entry.
typedef uint64_t set_t;

typedef struct sets_t_ {
  int n, k;
  int size;
} sets_t;

// Maximum number of empty spots handled by the midgame solvers: roots with at least 36-20 = 16 stones
#define MID_MAX_SPOTS 20

// List empty spots as bit indices into side_t
typedef struct empty_t_ {
  uint8_t empty[MID_MAX_SPOTS];
} empty_t;
