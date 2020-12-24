// Internal routines for midengine.cc and metal equivalent
#pragma once

#include "subsets_c.h"
#include "halfsuper_c.h"
#include "../base/board_c.h"
#include "../high/board_c.h"

typedef struct grab_t_ {
  int ny, lo;
  int size;
} grab_t;

// Constant information for an entire midsolve computation
typedef struct info_t_ {
  high_board_s root;
  int spots;
  empty_t empty;
  grab_t spaces[18+2];  // output = spaces[n], input = spaces[n+1]

  // Sums of sizes of temporary arrays up to each n
  int sets0_offsets[18+2];
  int sets1p_offsets[18+2];
  int cs1ps_offsets[18+2];
  int wins1_offsets[18+2];
} info_t;

// Constant information for an entire transposed midsolve
typedef struct transposed_t_ {
  high_board_s root;
  int spots;
  empty_t empty;
  grab_t spaces[18+2];  // output = spaces[n], input = spaces[n+1]

  // Sums of sizes of temporary arrays up to each n
  int sets0_offsets[18+2];
  int sets1_offsets[18+2];
  int cs0ps_offsets[18+2];
} transposed_t;

// Information needed for inner
typedef struct inner_t_ {
  int n, spots;
  int sets1_size;
  int sets1p_size;
  int sets0_offset, sets1p_offset, cs1ps_offset, wins1_offset;
  grab_t input, output;
} inner_t;

// Information needed for transposed which depends on n
typedef struct transposed_inner_t_ {
  int spots, n;
  uint16_t grid[2];  // grid[0] threadgroups of size grid[1]
  grab_t input, output;
  int sets0_offset, sets1_offset, cs0ps_offset;
} transposed_inner_t;

// Everything that's a function of just s0 in the double loop in midsolve_loop
typedef struct set0_info_t_ {
  halfsuper_s wins0;
  uint16_t child_s0s[18];
  uint16_t offset0[90];
  uint8_t empty1[18];
} set0_info_t;

typedef struct wins1_t_ {
  halfsuper_s after, before;
} wins1_t;

// Everything that's a function of just s1 for transposed
typedef struct set1_info_t_ {
  wins1_t wins1;
  uint16_t s1;
  uint16_t offset1p[90];
  uint8_t empty0[18];
} set1_info_t;

_Static_assert(sizeof(wins1_t) == 32, "");
_Static_assert(sizeof(set0_info_t) == 256, "");
