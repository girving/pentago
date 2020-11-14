// C interface to high_board_t, since Swift can't interoperate with C++.
#pragma once

#include "../pentago/mid/halfsuper_c.h"
#include "../pentago/mid/internal_c.h"
#include "../pentago/high/board_c.h"
#if __cplusplus
extern "C" {
#endif

typedef struct {
  high_board_s board;
  int value;
} high_board_value_t;

bool board_done(high_board_s b);
int board_immediate_value(high_board_s b);
high_board_s board_place_bit(high_board_s b, int bit);
high_board_s board_place_xy(high_board_s b, int x, int y);
high_board_s board_rotate(high_board_s b, int q, int d);

// Internal midsolver machinery
int board_workspace_size(high_board_s b);
info_t make_info_t(high_board_s b, int workspace_size);
inner_t make_inner_t(const info_t I, const int n);

// Returns number of entries
int board_midsolve_traverse(high_board_s b, const halfsupers_t* supers,
                            high_board_value_t results[1+18+8*18]);

// Returns number of entries
int board_midsolve(high_board_s b, high_board_value_t results[1+18+8*18]);

#if __cplusplus
}
#endif
