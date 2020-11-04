// Internal routines for midengine.cc and metal equivalent
#pragma once

#include "pentago/mid/subsets_c.h"
#include "pentago/mid/halfsuper_c.h"
#include "pentago/base/board_c.h"
#include "pentago/high/board_c.h"

typedef struct grab_t_ {
  int ny, lo;
  int size;
} grab_t;

// Constant information for an entire midsolve computation
typedef struct info_t_ {
  high_board_s root;
  int slice, spots;
  empty_t empty;
  grab_t spaces[18+2];  // output = spaces[n], input = spaces[n+1]

  // Sums of sizes of temporary arrays up to each n
  int sets0_offsets[18+2];
  int sets1p_offsets[18+2];
  int cs1ps_offsets[18+2];  // Rounded to even for alignment
  int wins_offsets[18+2];
} info_t;

// Information needed for inner
typedef struct inner_t_ {
  int n, spots, slice, k0, k1;
  sets_t sets1;
  int sets1p_size;
  grab_t input, output;
} inner_t;

// Everything that's a function of just s0 in the double loop in midsolve_loop
typedef struct set0_info_t_ {
  halfsuper_s wins0;
  uint16_t child_s0s[18];
  uint16_t offset0[90];
  uint8_t empty1[18];
} set0_info_t;

_Static_assert(sizeof(set0_info_t) == 256, "");
