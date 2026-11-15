// Exact values of boards with 18 or more stones, by a cascade of provers
//
// Most boards are cheap to prove, so we try cheap provers first and fall back to midsolve, whose
// cost is fixed (about 1.7 s natively for 18 stones) but which always finishes:
//
//   1. superengine forward search with iterative deepening, up to super_seconds.  Wins for the
//      player to move are sought at odd depths and losses at even depths, since a search to depth d
//      also covers every shorter line.
//   2. tie_prover_t: both players avoid losing, so the board is a tie.
//   3. midsolve.
//
// On uniformly random 18 stone boards with the default options, about 97% are settled before
// midsolve, at a mean cost of about 0.08 s per board.  The superengine stage checks its time only
// between depths, so it can overrun.  The superengine's transposition table is global; cascade_t
// initializes it if needed and never clears it, which is sound since its entries are exact.
#pragma once

#include "pentago/cascade/tie.h"
#include "pentago/high/board.h"
#include "pentago/mid/halfsuper.h"
#include "pentago/utility/array.h"
namespace pentago {

struct cascade_options_t {
  double super_seconds = 0.3;
  double tie_seconds = 1;
  int tie_depth = 4;
  int super_bits = 22;  // log2 size of the superengine's transposition table
};

enum class cascade_stage_t { superengine, tie, midsolve };

struct cascade_result_t {
  int value;  // For the player to move: 1 win, 0 tie, -1 loss
  cascade_stage_t stage;
};

class cascade_t {
 public:
  const cascade_options_t options;

  explicit cascade_t(const cascade_options_t& options = cascade_options_t());
  ~cascade_t();

  // The board must have at least 18 stones, not be in the middle of a move, and not be over
  cascade_result_t value(const high_board_t board);

  // The individual stages.  superengine returns 2 if unknown.
  int superengine(const high_board_t board, const double seconds);
  tie_prover_t& tie() { return tie_; }
  int midsolve(const high_board_t board);

 private:
  tie_prover_t tie_;
  int workspace_slice = 37;
  Array<halfsuper_s> workspace;
};

}  // namespace pentago
