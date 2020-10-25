// Check high level interface against forward search

#include "pentago/high/check.h"
#include "pentago/search/superengine.h"
NAMESPACE_PENTAGO

using std::max;

// If aggressive is true, true is win, otherwise true is win or tie.
static bool exact_lookup(const bool aggressive, const board_t board, const block_cache_t& cache) {
  {
    super_t wins;
    if (cache.lookup(aggressive,board,wins))
      return wins(0);
  }
  // Fall back to tree search
  const score_t score = super_evaluate(aggressive,100,board,Vector<int,4>());
  return (score&3)>=aggressive+1;
}

int value(const block_cache_t& cache, const high_board_t board) {
  if (board.done())
    return board.immediate_value();
  else if (!board.middle()) {
    // If we're not at a half move, look up the result
    const auto flip = flip_board(board.board(), board.turn());
    return exact_lookup(true,flip,cache)+exact_lookup(false,flip,cache)-1;
  } else {
    int best = -1;
    for (const auto& move : board.moves()) {
      best = max(best, -value(cache, move));
      if (best == 1)
        break;
    }
    return best;
  }
}

int value_check(const block_cache_t& cache, const high_board_t board) {
  GEODE_ASSERT(!board.middle());
  if (board.done())
    return board.immediate_value();
  {
    int value = -1;
    for (const auto& move : board.moves())
      value = max(value, pentago::value(cache, move));
    const int lookup = pentago::value(cache, board);
    if (value != lookup)
      THROW(AssertionError, "high_board_t.value_check: board %lld, turn %d, middle %d, computed %d != lookup %d",
            board.board(), board.turn(), board.middle(), value, lookup);
    return value;
  }
}

Vector<int,3> sample_check(const block_cache_t& cache, RawArray<const board_t> boards,
                           RawArray<const Vector<super_t,2>> wins) {
  GEODE_ASSERT(boards.size() == wins.size());
  Vector<int,3> counts;
  Random random(152311341);
  for (const int i : range(boards.size())) {
    const int r = random.bits<uint8_t>();
    const int n = count_stones(boards[i]);
    const auto board = high_board_t::from_board(transform_board(symmetry_t(local_symmetry_t(r)), boards[i]), false);
    const int value = value_check(cache, board);
    GEODE_ASSERT(value == wins[i][n&1](r)-wins[i][!(n&1)](r));
    counts[value+1]++;
  }
  return counts;
}

END_NAMESPACE_PENTAGO
