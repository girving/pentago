// Core tree search engine
#pragma once

#include "pentago/base/board.h"
#include "pentago/base/score.h"
namespace pentago {

// Evaluate a position down to a certain game tree depth.  If the depth is exceeded, assume a tie.
// We maintain the invariant that player 0 is always the player to move.  The result is 1 for a win
// 0 for tie, -1 for loss.
score_t evaluate(int depth, board_t board);

// Evaluate a position down to a certain game tree depth, ignoring rotations and white wins.
score_t simple_evaluate(int depth, board_t board);

}
