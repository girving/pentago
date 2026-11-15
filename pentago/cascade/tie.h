// Tie proofs via the undo strategy
//
// A player X who always places a stone and then undoes the opponent's last rotation keeps the
// quadrants fixed in one frame.  In that frame the game becomes static: the opponent (the Maker)
// places a stone and must rotate, and wins if its stones cover a five-in-a-row line pulled back
// through at most one rotation; X (the Breaker) places anywhere.  If X wins this static game, X does
// not lose the real game.  Proving this for both players proves a tie.
//
// In the static game the Maker may not pick a rotation after which X wins or ties at once, a Maker
// rotation that also completes a five for X is a tie, and X replies only where it stops the Maker's
// next immediate win.  Other ties are counted against the Maker, so every claim is sound.
//
// Pure undo from the current position is rarely enough at 18 stones, so a player may also answer
// up to `depth` opponent moves with free moves of its own (any placement and rotation) before
// switching to undo.  The player to move additionally gets its first move.  The static game is
// solved with bit-sliced line counts (see tie.cc); the outer search over real moves uses a history
// heuristic for move ordering.
//
// All memo entries are exact and depend only on the position, so the prover keeps them across
// boards and calls.  The code avoids std containers and exceptions so that it can be compiled for
// freestanding wasm; there the clock must be supplied with set_clock.
#pragma once

#include "pentago/base/board.h"
#include "pentago/high/board.h"
namespace pentago {

struct tie_result_t {
  // Fewest free replies with which each player was shown not to lose, or -1 if not shown
  int mover = -1, other = -1;
  bool timeout = false;

  bool tie() const { return mover >= 0 && other >= 0; }
};

// Seconds since an arbitrary origin
typedef double (*tie_clock_t)();

class tie_prover_t {
 public:
  // The static game memo has 2^memo_bits 8-byte entries, and the outer search memo 2^(memo_bits-3)
  // 16-byte entries
  explicit tie_prover_t(const int memo_bits = 20);
  ~tie_prover_t();
  tie_prover_t(const tie_prover_t&) = delete;
  void operator=(const tie_prover_t&) = delete;

  // Try to show that each player avoids losing, with up to max_depth free replies, stopping after
  // the given number of seconds.  A timed out search claims nothing it has not finished.
  // The board must not be in the middle of a move, and the game must not be over.
  tie_result_t prove(const high_board_t board, const int max_depth, const double seconds);

  // Does the player not to move avoid losing by pure undo from here?  For testing.
  bool undo_holds(const high_board_t board);

  // For use inside other searches, on raw sides: does the player to move (or the other player) hold
  // with up to depth free replies?  False on timeout.
  bool mover_holds(const side_t to_move, const side_t other, const int depth, const double seconds);
  bool other_holds(const side_t to_move, const side_t other, const int depth, const double seconds);

  // Forget all memoized results
  void clear();

  // Replace the clock used for timeouts.  Natively the default is wall time; in wasm there is no
  // default, and without a clock searches never time out.
  void set_clock(const tie_clock_t clock);

  // For testing: check the time every 2^bits search nodes (default 10), and verify incremental state
  // (memo keys and line counts) against scratch recomputation at every node
  void set_time_check_bits(const int bits);
  void set_paranoid(const bool paranoid);

 private:
  struct impl_t;
  impl_t* const impl;
};

}  // namespace pentago
