#include "pentago/base/board.h"
#include "pentago/base/score.h"
#include "pentago/base/symmetry.h"
#include "gtest/gtest.h"
namespace pentago {
namespace {

TEST(slow, won) {
  Random random(7);
  for (int i = 0; i < (1<<20); i++) {
    const auto side = random_side(random);
    ASSERT_EQ(won(side), slow_won(side));
  }
}

}  // namespace
}  // namespace pentago
