#include "pentago/base/board.h"
#include "pentago/base/count.h"
#include "pentago/base/hash.h"
#include "pentago/base/moves.h"
#include "pentago/base/score.h"
#include "pentago/base/superscore.h"
#include "pentago/base/symmetry.h"
#include "pentago/utility/range.h"
#include "pentago/utility/threefry.h"
#include "pentago/utility/log.h"
#include "gtest/gtest.h"
#include <unordered_set>
namespace pentago {
namespace {

using std::function;
using std::get;
using std::make_tuple;
using std::tuple;
using std::unordered_set;

Array<board_t> old_moves(const board_t board, const bool turn, const bool simple) {
  const auto flipped = flip_board(board, turn);
  const auto results = simple ? simple_moves(flipped) : moves(flipped);
  for (auto& move : results) move = flip_board(move, !turn);
  return results;
}

TEST(pentago, misc) {
  Random random(73120);
  for (int step = 0; step < 100; step++) {
    const auto board = random_board(random);
    ASSERT_EQ(pack(unpack(board, 0), unpack(board, 1)), board);
    ASSERT_EQ(from_table(to_table(board)), board);
    ASSERT_EQ(flip_board(flip_board(board)), board);
  }
}

template<class A> auto choose(Random& random, const A& options) {
  GEODE_ASSERT(options.size());
  return options[random.uniform(options.size())];
}

board_t slow_rotate(const board_t board, const int qx, const int qy, int count) {
  GEODE_ASSERT(range(2).contains(qx) && range(2).contains(qy));
  auto table = to_table(board);
  for (const int i __attribute__((unused)) : range(count & 3)) {
    auto copy = table.copy();
    for (const int x : range(3)) {
      for (const int y : range(3)) {
        copy(3*qx+x,3*qy+y) = table(3*qx+y,3*qy+2-x);
      }
    }
    table = copy;
  }
  return from_table(table);
}

void win_test(const string& rotations) {
  Random random(81382);
  function<int(board_t)> status;
  function<int(side_t,side_t)> side_closeness;
  const auto closeness = [&side_closeness](board_t board) {
    check_board(board);
    return side_closeness(unpack(board, 0), unpack(board, 1)); 
  };
  const auto distance = [closeness](const board_t board) {
    return 6 - (closeness(board) >> 16);
  };
  const auto log = [rotations, distance, closeness](const string& name, const board_t board) {
    const int close = closeness(board);
    slog("rotations %s, %s: closeness %d, distance %d, count %d",
         rotations, name, close, distance(board), close & 0xffff);
  };
  unordered_set<board_t> special;
  if (rotations == "none") {
    status = pentago::status;
    side_closeness = unrotated_win_closeness;
  } else if (rotations == "single") {
    status = [](board_t board) {
      const auto st = rotated_won(unpack(board, 0));
      bool won = false;
      for (const int qx : range(2))
        for (const int qy : range(2))
          for (const int count : {-1, 1})
            won |= pentago::status(slow_rotate(board, qx, qy, count)) & 1;
      GEODE_ASSERT(st == won);
      return st;
    };
    side_closeness = rotated_win_closeness;
    // In some cases, black can win by rotating the same quadrant in either direction.
    // Closeness computation counts this as a single "way", even though it is really two,
    // which may make it impossible to white to reduce closeness.
    special = {3936710634889235564, 8924889845, 53214872776869257, 616430895238742070,
               45599372358462108, 1194863002822977710, 3559298137047367680, 2485989107724386484,
               1176595961439925315};
  } else if (rotations == "any") {
    status = arbitrarily_rotated_status;
    side_closeness = arbitrarily_rotated_win_closeness;
  } else {
    GEODE_ASSERT(false);
  }
  int wins = 0, ties = 0;
  for (const int i : range(100)) {
    // Start with an empty board
    board_t board = 0;
    ASSERT_EQ(distance(board), 5);
    log("\nstart", board);
    if (rotations == "none") {
      // Give black several free moves to even the odds
      for (const int _ __attribute__((unused)) : range(random.uniform(15)))
        board = choose(random, old_moves(board, 0, true));
      log("seeds", board);
    }
    while (distance(board)) {
      // Verify that a well-placed white stone reduces closeness
      try {
        if (rotations != "any") {  // We should almost always be able to find a move reducing closeness
          bool found = false;
          for (const auto b : old_moves(board, 1, true)) {
            ASSERT_NE(b, board);
            if (closeness(b) < closeness(board)) {
              found = true;
              board = b;
              break;
            }
          }
        } else {  // Black has too much freedom to rotate, so don't require a reduction in closeness
          board_t best = 0;
          const auto signature = [closeness](board_t b) {
            return make_tuple(closeness(b), hash_board(b));
          };
          for (const auto b : old_moves(board, 1, true)) {
            ASSERT_NE(b, board);
            if (!best || signature(b) < signature(best))
              best = b;
          }
          board = best;  
        }
      } catch (const AssertionError&) {
        if (!contains(special, board)) {
          const int close = closeness(board);
          slog("i %d, board %d, distance %d, ways %d\n%s",
               i, board, distance(board), close & 0xffff, str_board(board));
          throw;
        }
      }
      log("after white", board);
      if (distance(board) == 6) {
        ties += 1;
        break;
      }
      // Verify that a well-placed black stone reduces distance
      vector<board_t> options;
      for (const auto b : old_moves(board, 0, true))
        if (distance(b) < distance(board))
          options.push_back(b);
      ASSERT_GT(options.size(), 0);
      board = choose(random, options);
      log("after black", board);
      // Verify that we've won iff distance==0
      ASSERT_EQ(distance(board)==0, status(board)&1);
      if (distance(board) == 0) {
        wins += 1;
        break;
      }
    }
  }
  slog("wins %d, ties %d", wins, ties);
}

TEST(pentago, unrotated_win) {
  win_test("none");
}

TEST(pentago, rotated_win) {
  win_test("single");
}

TEST(pentago, arbitrarily_rotated_win) {
  win_test("any");
}

TEST(pentago, hash) {
  // Verify that hash_board and inverse_hash_board are inverses
  Random random(71231);
  for (int i = 0; i < 256; i++) {
    const auto key = random.bits<uint64_t>();
    ASSERT_EQ(key, inverse_hash_board(hash_board(key)));
  }

  // Verify that no valid board maps to 0
  try {
    check_board(inverse_hash_board(0));
    GEODE_ASSERT(false);
  } catch (const ValueError&) {
    // Squash error
  }
}

TEST(pentago, popcounts_over_stabilizers) {
  const int steps = 1024;
  Random random(83191941);
  unordered_set<board_t> seen;
  for (int step=0;step<steps;step++) {
    // Generate a board with significant probability of duplicate quadrants
    const int stones = random.uniform<int>(0,37);
    board_t board = random_board(random,stones);
    const int qs = random.uniform<int>(0,256);
    board = quadrants(quadrant(board,qs>>0&3),
                      quadrant(board,qs>>2&3),
                      quadrant(board,qs>>4&3),
                      quadrant(board,qs>>6&3));
    // Compute meaningless data
    Vector<super_t,2> wins;
    wins[0] = super_meaningless(board);
    wins[1] = ~wins[0];
    // Count the fast way
    const Vector<uint16_t,3> fast = popcounts_over_stabilizers(board,wins);
    ASSERT_EQ(fast[0] + fast[1], fast[2]);
    // Count the slow way 
    Vector<uint16_t,3> slow;
    seen.clear();
    for (int g=0;g<8;g++)
      for (int l=0;l<256;l++) {
        const board_t b = transform_board(symmetry_t(g,l),board);
        for (int q=0;q<4;q++)
          if (get<0>(rotation_standardize_quadrant(quadrant(board,q))) !=
              get<0>(rotation_standardize_quadrant(quadrant(b,q))))
            goto skip;
        if (seen.insert(b).second)
          slow[0] += meaningless(b);
        skip:;
      }
    slow[2] = seen.size();
    slow[1] = slow[2] - slow[0];
    ASSERT_EQ(slow, fast);
  }
}

}  // namespace
}  // hamespace pentago
