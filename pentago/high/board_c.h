// High level interface to board
//
// high_board_t wraps a nice class interface around the bare uint64_t that is board_t.
// It also splits the underlying place+rotation ply into two separate plys, which
// corresponds more closely to the final frontend interface to the game.
//
// When converted to a string, boards with the player to move about to place a stone
// are represented as a bare int; boards with the player to move about to rotate have
// a trailing 'm'.  For example, 1m.
#pragma once

#include "pentago/base/board_c.h"

typedef struct high_board_s_ {
  uint64_t side_[2];  // black, white (first player, second player)
  uint32_t ply_;
} high_board_s;
