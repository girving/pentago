// Exact values of boards with 18 or more stones, by a cascade of provers
//
// Most boards are cheap to prove, so we try cheap provers first and fall back to midsolve, whose
// cost is fixed (about 1.7 s natively for 18 stones) but which always finishes:
//
//   1. Rotation abstracted forward search (halfengine_t, a halfsuper_t port of the superengine)
//      with iterative deepening, up to super_seconds.  Wins for the player to move are sought at
//      odd depths and losses at even depths, since a search to depth d also covers every shorter
//      line.
//   2. tie_prover_t: both players avoid losing, so the board is a tie.
//   3. midsolve.
//
// On uniformly random 18 stone boards with the default options, about 97% are settled before
// midsolve, at a mean cost of about 0.08 s per board.  The superengine stage checks its time only
// between depths, so it can overrun.  The superengine's transposition table is global; cascade_t
// initializes it if needed and keeps it across boards, which is sound since its entries are exact.
#pragma once

#include "pentago/cascade/halfengine.h"
#include "pentago/cascade/tie.h"
#include "pentago/high/board.h"
#include "pentago/mid/midengine.h"
#include "pentago/mid/halfsuper.h"
#include "pentago/utility/array.h"
namespace pentago {

struct cascade_options_t {
  double super_seconds = 0.3;
  double tie_seconds = 1;
  int tie_depth = 4;
  int super_bits = 22;  // log2 entries in the forward search's transposition table (40 bytes each)
  // Cheap stage budget per placement, after which it goes to midsolve.  Unlimited by default: on real
  // 17 stone positions a 1 s budget sent more placements to midsolve and was slower overall.
  double placement_seconds = 1e9;
  bool cutoffs = true;  // Tie prover cutoffs inside the forward search (see halfengine_cutoffs_t)
  int cutoff_depth = 1;  // Free replies for cutoff tie prover calls (best of 0, 1, 2 on real positions)

  // For children(): placements left unknown by the cheap stages that get a dense midsolve each, and
  // the time limit for the cheap stages over all placements.  Past either, children() gives up.
  int max_dense = 1;
  double children_seconds = 4;
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

  // Values of the 8 rotations of a placed board (in the middle of a move), for the player to move
  // after the rotation.  Rotation s turns quadrant s/2 left if s is even, right if odd, matching
  // midsolve's order.  Each rotation goes through the cheap stages until options.placement_seconds
  // runs out; if any remains unknown, one midsolve of the placed board settles all of them.
  void placement(const high_board_t placed, cascade_result_t results[8]);

  // Values of a board with at least 17 stones, not in the middle of a move, together with every
  // placement and every rotation of every placement, in midsolve's order (see midsolve_traverse).
  // Returns false, with results incomplete, if the cheap stages leave more than max_dense placements
  // unknown or run past children_seconds: then the board needs a full midsolve (e.g. the tiled
  // solver), which costs about the same for every board.
  bool children(const high_board_t board, mid_values_t& results);

  // The individual stages.  superengine returns 2 if unknown.
  int superengine(const high_board_t board, const double seconds);
  tie_prover_t& tie() { return tie_; }
  halfengine_t& forward() { return engine; }
  int midsolve(const high_board_t board);

 private:
  halfengine_t engine;
  tie_prover_t tie_;
  RawArray<halfsuper_s> workspace;  // For midsolves of boards with at least 18 stones, made lazily
#ifndef __wasm__
  Array<halfsuper_s> workspace_owner;
#endif

  void midsolve(const high_board_t board, mid_values_t& results);
  bool cheap_placement(const high_board_t placed, cascade_result_t results[8], const double start,
                       const double seconds);
  void finish_placement(const high_board_t placed, cascade_result_t results[8]);
};

}  // namespace pentago
