// Exact values of boards with 18 or more stones, by a cascade of provers

#include "pentago/cascade/cascade.h"
#include "pentago/base/score.h"
#include "pentago/mid/midengine.h"
#include "pentago/search/superengine.h"
#include "pentago/search/supertable.h"
#include "pentago/utility/debug.h"
#include "pentago/utility/wall_time.h"
namespace pentago {

cascade_t::cascade_t(const cascade_options_t& options)
  : options(options) {
  if (supertable_bits() < options.super_bits)
    init_supertable(options.super_bits, false);
}

cascade_t::~cascade_t() {}

static int superengine_search(const high_board_t board, const double seconds) {
  // The superengine wants the player to move as side 0
  const board_t b = flip_board(board.board(), board.turn());
  const int full = 36 - board.count();
  const auto start = wall_time();
  for (int d = 1; d <= full; d++) {
    // Wins for the player to move complete on its own moves (odd depths) and losses on the
    // opponent's (even depths), up to rotations that complete a line for the other side; a search to
    // depth d covers all shorter lines, so those are found one depth later.  At full depth we need
    // both searches to see a tie.
    bool not_win = false;
    if (d & 1 || d == full) {
      if ((super_evaluate(true, d, b, Vector<int,4>()) & 3) == 2) return 1;
      not_win = true;
    }
    if (!(d & 1) || d == full) {
      if ((super_evaluate(false, d, b, Vector<int,4>()) & 3) == 0) return -1;
      if (d == full && not_win) return 0;
    }
    if ((wall_time() - start).seconds() > seconds) break;
  }
  return 2;
}

int cascade_t::superengine(const high_board_t board, const double seconds) {
  GEODE_ASSERT(!board.middle() && !board.done());
  return superengine_search(board, seconds);
}

int cascade_t::midsolve(const high_board_t board) {
  GEODE_ASSERT(!board.middle() && board.count() >= 18);
  if (board.count() < workspace_slice) {
    workspace_slice = board.count();
    workspace = midsolve_workspace(workspace_slice);
  }
  const auto values = pentago::midsolve(board, workspace);
  for (const auto& [raw, v] : values)
    if (raw == board.raw()) return v;
  GEODE_ASSERT(false, "midsolve did not return the root");
}

cascade_result_t cascade_t::value(const high_board_t board) {
  GEODE_ASSERT(!board.middle() && !board.done() && board.count() >= 18);
  const int s = superengine(board, options.super_seconds);
  if (s != 2) return {s, cascade_stage_t::superengine};
  if (tie_.prove(board, options.tie_depth, options.tie_seconds).tie()) return {0, cascade_stage_t::tie};
  return {midsolve(board), cascade_stage_t::midsolve};
}

}  // namespace pentago
