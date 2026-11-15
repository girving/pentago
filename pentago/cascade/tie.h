// Tie proofs via the undo strategy
//
// A player X who always places a stone and then undoes the opponent's last rotation keeps the
// quadrants fixed in one frame.  In that frame the game becomes static: the opponent (the Maker)
// places a stone and must rotate, and wins if its stones cover a five-in-a-row line pulled back
// through at most one rotation; X (the Breaker) places anywhere.  If X wins this static game, X does
// not lose the real game.  Proving this for both players proves a tie.
//
// In the static game the Maker may not pick a rotation after which X wins or ties at once, and X
// replies only where it stops the Maker's next immediate win.  Ties are always counted against the
// Maker, so every claim is sound.
//
// Pure undo from the current position is rarely enough at 18 stones, so a player may also answer
// up to `depth` opponent moves with free moves of its own (any placement and rotation) before
// switching to undo.  The player to move additionally gets its first move.  The static game is
// solved with bit-sliced line counts (see tie.cc); the outer search over real moves uses a history
// heuristic for move ordering.
#pragma once

#include "pentago/base/board.h"
#include "pentago/high/board.h"
#include <memory>
namespace pentago {

struct tie_result_t {
  // Fewest free replies with which each player was shown not to lose, or -1 if not shown
  int mover = -1, other = -1;
  bool timeout = false;

  bool tie() const { return mover >= 0 && other >= 0; }
};

class tie_prover_t {
 public:
  // The static game memo has 2^memo_bits 8-byte entries
  explicit tie_prover_t(const int memo_bits = 20);
  ~tie_prover_t();

  // Try to show that each player avoids losing, with up to max_depth free replies, stopping after
  // the given number of seconds.  A timed out search claims nothing it has not finished.
  // The board must not be in the middle of a move, and the game must not be over.
  tie_result_t prove(const high_board_t board, const int max_depth, const double seconds);

  // Does the player not to move avoid losing by pure undo from here?  For testing.
  bool undo_holds(const high_board_t board);

  // For testing: check the time every 2^bits search nodes (default 10), and verify incremental state
  // (memo keys and line counts) against scratch recomputation at every node
  void set_time_check_bits(const int bits);
  void set_paranoid(const bool paranoid);

 private:
  struct impl_t;
  const std::unique_ptr<impl_t> impl;
};

}  // namespace pentago
