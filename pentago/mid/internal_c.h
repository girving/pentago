// Internal routines for midengine.cc and metal equivalent
#pragma once

#include "pentago/mid/subsets_c.h"
#include "pentago/mid/halfsuper_c.h"
#include "pentago/base/board_c.h"
#include "pentago/high/board_c.h"

typedef struct grab_t_ {
  int ny, lo;
} grab_t;

// Constant information for midsolve_loop
typedef struct info_t_ {
  high_board_s root;
  int n, parity, slice, spots, k0, k1;
  bool done;
  side_t root0, root1;
  sets_t sets0, sets1, sets1p, sets0_next;
  int cs1ps_size;
  empty_t empty;
  grab_t input, output;
} info_t;

typedef struct wins_info_t_ {
  empty_t empty;
  side_t root0, root1;
  sets_t sets1, sets0_next;
  int size;
  bool parity;
} wins_info_t;

// Everything that's a function of just s0 in the double loop in midsolve_loop
typedef struct set0_info_t_ {
  side_t side0;
  set_t set0;
  uint32_t filled0;
  halfsuper_s wins0;
  halfsuper_s child_wins0[18];
  uint8_t empty1[18];
  uint16_t child_s0s[18];
  uint16_t offset1[9][18];
  uint16_t offset0[9][18];
} set0_info_t;

_Static_assert(sizeof(set0_info_t) >= 1032, "");
_Static_assert(sizeof(set0_info_t) <= 1048, "");
