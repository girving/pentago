#include "pentago/base/board.h"
#include "pentago/base/symmetry.h"
#include "gtest/gtest.h"
namespace pentago {
namespace {

TEST(slow, board) {
  Random random(7);
  for (int i = 0; i < 256; i++) {
    const auto board = random_board(random);
    const auto side0 = unpack(board, 0);
    const auto side1 = unpack(board, 1);
    const auto [slow0, slow1] = slow_unpack(board);
    ASSERT_EQ(side0, slow0);
    ASSERT_EQ(side1, slow1);
    ASSERT_EQ(board, slow_pack(side0, side1));
    ASSERT_EQ(count_stones(board), slow_count_stones(board));
    for (const int turn : range(2))
      ASSERT_EQ(flip_board(board, turn), slow_flip_board(board, turn));
  }
}

TEST(slow, transform_board) {
  Random random(7);
  for (int i = 0; i < 256; i++) {
    const auto s = random_symmetry(random);
    const auto board = random_board(random);
    ASSERT_EQ(transform_board(s, board), slow_transform_board(s, board));
  }
}

}  // namespace
}  // namespace pentago
