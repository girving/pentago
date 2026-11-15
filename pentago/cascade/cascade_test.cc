// Tests for the cascade of provers
//
// Exact values come from a table of stored positions, generated once with midsolve (see
// gen_cascade_cases.cc).  Stages are tested directly, since which stage
// settles a board depends on timing.

#include "pentago/cascade/cascade.h"
#include "pentago/search/supertable.h"
#include "pentago/utility/test_assert.h"
#include "gtest/gtest.h"
namespace pentago {
namespace {

// A stored position: raw board, exact value for the player to move, and the stage that settled it
// when generated (0 superengine, 1 tie prover, 2 midsolve)
struct cascade_case_t { raw_t raw; int value; int stage; };

const cascade_case_t cascade_cases[] = {
  {4143369433081715723ull, 1, 0},  // 18 stones, 0.000 s
  {1580201644851798870ull, 1, 0},  // 18 stones, 0.001 s
  {3832359551168169375ull, 0, 1},  // 18 stones, 0.324 s
  {618155556552381424ull, 0, 1},  // 18 stones, 0.372 s
  {4812393538570422200ull, -1, 0},  // 19 stones, 0.004 s
  {3551985035510943311ull, 1, 0},  // 19 stones, 0.000 s
  {4700654453631164017ull, 0, 1},  // 19 stones, 0.451 s
  {1341791948651503660ull, 0, 1},  // 19 stones, 0.414 s
  {1566197779235144708ull, 1, 0},  // 20 stones, 0.000 s
  {966043921828873575ull, 1, 0},  // 20 stones, 0.000 s
  {2118739575082008327ull, 0, 1},  // 20 stones, 0.434 s
  {2449678874809936727ull, 0, 1},  // 20 stones, 0.492 s
  {4009637019409199954ull, 1, 0},  // 21 stones, 0.000 s
  {2122906522422951430ull, 1, 0},  // 21 stones, 0.000 s
  {279010413707801660ull, 0, 1},  // 21 stones, 0.310 s
  {56628732504975894ull, 0, 1},  // 21 stones, 0.397 s
  {2270730552328454768ull, 0, 0},  // 22 stones, 0.156 s
  {1872689219147482413ull, 1, 0},  // 22 stones, 0.000 s
  {1527635969413106211ull, 1, 0},  // 23 stones, 0.000 s
  {3026205318734422387ull, 1, 0},  // 23 stones, 0.000 s
  {1619947302200484953ull, 1, 0},  // 24 stones, 0.000 s
  {273634333888494286ull, 0, 0},  // 24 stones, 0.001 s
  {3480797873212361722ull, 1, 0},  // 25 stones, 0.000 s
  {4480831658209384566ull, 0, 0},  // 25 stones, 0.002 s
  {4295087560213080432ull, 1, 0},  // 26 stones, 0.000 s
  {2954709034208274521ull, 1, 0},  // 26 stones, 0.000 s
};

// Children of the 17 stone board 217635266476703744 (from a real game), in the order a 17 stone
// client search visits them.  Searching them in turn with the superengine's table kept across
// searches used to make super_store throw a false positive consistency error.
const cascade_case_t reused_table_cases[] = {
  {217635266476703762ull, 0, 0},
  {217635266476705202ull, 0, 0},
  {217635266081914882ull, 0, 0},
  {217635265762623490ull, 0, 0},
  {217595718417842178ull, 0, 0},
  {217648976012312578ull, 1, 0},
  {1989801719846993922ull, 1, 0},
  {2888269845507407874ull, 0, 0},
  {217635266476704230ull, 0, 0},
};

TEST(cascade, reused_table) {
  cascade_t cascade;
  clear_supertable();
  for (const auto& c : reused_table_cases) {
    const auto board = high_board_t::from_raw(c.raw);
    SCOPED_TRACE(board.name());
    const int s = cascade.superengine(board, cascade.options.super_seconds);
    if (s != 2) {
      PENTAGO_ASSERT_EQ(s, c.value);
    }
  }
}

// The full cascade gets every value right
TEST(cascade, values) {
  cascade_t cascade;
  for (const auto& c : cascade_cases) {
    const auto board = high_board_t::from_raw(c.raw);
    SCOPED_TRACE(board.name());
    PENTAGO_ASSERT_EQ(cascade.value(board).value, c.value);
  }
}

// Each stage on its own: the superengine without a time limit is exact, the tie prover proves the
// ties it settled, and midsolve agrees everywhere it is cheap
TEST(cascade, stages) {
  cascade_t cascade;
  int supers = 0, ties = 0, mids = 0;
  for (const auto& c : cascade_cases) {
    const auto board = high_board_t::from_raw(c.raw);
    SCOPED_TRACE(board.name());
    if (c.stage == 0) {
      PENTAGO_ASSERT_EQ(cascade.superengine(board, 1e9), c.value);
      supers++;
    } else if (c.stage == 1) {
      PENTAGO_ASSERT_EQ(c.value, 0);
      ASSERT_TRUE(cascade.tie().prove(board, cascade.options.tie_depth, 1e9).tie());
      ties++;
    }
    if (board.count() >= 22) {
      PENTAGO_ASSERT_EQ(cascade.midsolve(board), c.value);
      mids++;
    }
  }
  PENTAGO_ASSERT_GE(supers, 10);
  PENTAGO_ASSERT_GE(ties, 6);
  PENTAGO_ASSERT_GE(mids, 6);
}

// With no time for the cheap stages, everything falls through to midsolve
TEST(cascade, fallback) {
  cascade_options_t options;
  options.super_seconds = options.tie_seconds = 0;
  cascade_t cascade(options);
  int fallbacks = 0;
  for (const auto& c : cascade_cases) {
    const auto board = high_board_t::from_raw(c.raw);
    if (board.count() < 22) continue;
    SCOPED_TRACE(board.name());
    const auto r = cascade.value(board);
    PENTAGO_ASSERT_EQ(r.value, c.value);
    fallbacks += r.stage == cascade_stage_t::midsolve;
  }
  // Depth one superengine searches still settle some boards
  PENTAGO_ASSERT_GE(fallbacks, 1);
}

}  // namespace
}  // namespace pentago
