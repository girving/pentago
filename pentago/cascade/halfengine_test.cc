// Tests for the halfsuper forward search, against the superengine

#include "pentago/cascade/halfengine.h"
#include "pentago/cascade/tie.h"
#include "pentago/base/score.h"
#include "pentago/search/superengine.h"
#include "pentago/search/supertable.h"
#include "pentago/utility/random.h"
#include "pentago/utility/test_assert.h"
#include "gtest/gtest.h"
namespace pentago {
namespace {

TEST(halfengine, closeness) {
  Random random(1831);
  for (int i = 0; i < 100000; i++) {
    const board_t board = random_board(random, random.uniform<int>(0, 37));
    const side_t black = unpack(board, 0), white = unpack(board, 1);
    PENTAGO_ASSERT_EQ(halfengine_t::closeness(black, white), arbitrarily_rotated_win_closeness(black, white));
  }
}

TEST(halfengine, wins) {
  Random random(1832);
  int nonzero = 0;
  for (int i = 0; i < 100000; i++) {
    const side_t side = unpack(random_board(random, random.uniform<int>(0, 37)), 0);
    for (const bool parity : {false, true}) {
      const halfsuper_t w = halfengine_t::fast_wins(side, parity);
      ASSERT_TRUE(w == halfsuper_wins(side, parity)) << side << " " << parity;
      for (int j = 0; j < 128; j += 7)
        PENTAGO_ASSERT_EQ(won(halfengine_t::rotate_state(side, parity, j)), w[j]);
      nonzero += bool(w);
    }
  }
  PENTAGO_ASSERT_GT(nonzero, 10000);
}

// Every depth and both modes agree with the superengine, with or without tie prover cutoffs (which
// give facts about the whole game, so don't change depth limited answers).  The superengine's table is cleared for
// each board, but ours is shared across all of them, which also tests reuse.
static void superengine_test(const bool cutoffs) {
  init_supertable(20, false);
  halfengine_t engine(16);
  tie_prover_t prover;
  if (cutoffs) {
    engine.cutoffs.prover = &prover;
    engine.cutoffs.min_depth = 2;
    engine.cutoffs.max_states = 128;
  }
  Random random(7);
  int counts[2][2] = {{0, 0}, {0, 0}};
  for (int i = 0; i < (cutoffs ? 1000 : 3000); i++) {
    // The tie prover needs at least 18 stones
    const int stones = random.uniform<int>(cutoffs ? 18 : 12, 33);
    const board_t board = random_board(random, stones);
    const side_t side0 = unpack(board, 0), side1 = unpack(board, 1);
    if (won(side0) || won(side1)) continue;
    clear_supertable();
    for (int depth = 0; depth <= min(5, 36 - stones); depth++)
      for (const bool aggressive : {false, true}) {
        const bool ours = engine.evaluate(aggressive, depth, side0, side1);
        const bool theirs = (super_evaluate(aggressive, depth, board, Vector<int,4>()) & 3) == 1 + aggressive;
        PENTAGO_ASSERT_EQ(ours, theirs);
        counts[aggressive][ours]++;
      }
  }
  // Both answers occur in both modes, and cutoffs happen
  for (const int a : range(2))
    for (const int v : range(2))
      PENTAGO_ASSERT_GT(counts[a][v], 50);
  if (cutoffs) PENTAGO_ASSERT_GT(engine.cutoff_hits, 1000u);
}

TEST(halfengine, superengine) { superengine_test(false); }
TEST(halfengine, cutoffs) { superengine_test(true); }

// Late boards searched to the end: exact values agree with the superengine
TEST(halfengine, search) {
  init_supertable(20, false);
  halfengine_t engine(18);
  Random random(11);
  int values[3] = {0, 0, 0};
  const auto clock = []() { return 0.0; };
  for (int i = 0; i < 2000; i++) {
    const board_t board = random_board(random, random.uniform<int>(26, 34));
    const side_t side0 = unpack(board, 0), side1 = unpack(board, 1);
    if (won(side0) || won(side1)) continue;
    const int v = engine.search(side0, side1, 1e9, clock);
    const int full = 36 - popcount(side0 | side1);
    const bool win = (super_evaluate(true, full, board, Vector<int,4>()) & 3) == 2;
    const bool notlose = (super_evaluate(false, full, board, Vector<int,4>()) & 3) == 1;
    PENTAGO_ASSERT_EQ(v, win ? 1 : notlose ? 0 : -1);
    values[v + 1]++;
  }
  for (const int n : values) PENTAGO_ASSERT_GT(n, 10);
}

}  // namespace
}  // namespace pentago
