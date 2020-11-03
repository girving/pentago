// Internal routines for midengine.cc and metal equivalent
#pragma once

#include "pentago/mid/subsets_c.h"
#include "pentago/mid/halfsuper_c.h"
#include "pentago/base/board_c.h"
#include "pentago/high/board_c.h"

typedef struct grab_t_ {
  int ny, lo;
} grab_t;

// Constant information for an entire midsolve computation
typedef struct info_t_ {
  high_board_s root;
  int slice, spots;
  empty_t empty;
  grab_t spaces[18+2];  // output = spaces[n], input = spaces[n+1]
  int sets1p_offsets[18+2];  // sets1p_offsets[n] = sum_{k < n} sets1p(n).size
  int cs1ps_offsets[18+2];  // cs1ps_offsets[n] = sum_{k < n} cs1ps_size[k]
  int wins_offsets[18+2];  // wins_offsets[n] = sum_{k < n} wins_size[k]
} info_t;

// Everything that's a function of just s0 in the double loop in midsolve_loop
typedef struct set0_info_t_ {
  halfsuper_s wins0;
  uint16_t child_s0s[18];
  uint16_t offset0[90];
  uint8_t empty1[18];
} set0_info_t;

_Static_assert(sizeof(set0_info_t) == 256, "");
