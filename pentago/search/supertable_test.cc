#include "pentago/search/supertable.h"
#include "pentago/utility/log.h"
#include "gtest/gtest.h"

namespace pentago {
namespace {

int choice(Random& random, int choices=2) {
  return random.uniform<int>(0,choices);
}

void supertable_test(const int epochs) {
  // Prepare
  const int bits = 10;
  const int count = 1<<bits;
  init_supertable(bits);
  Random random(98312);
  uint64_t total_lookups = 0;

  // Repeatedly clear the supertable and run through a bunch of operations
  Array<board_t> boards;
  for (int epoch=0;epoch<epochs;epoch++) {
    // Start fresh
    clear_supertable();
    int successful_lookups = 0;

    // Generate a bunch of random boards
    Array<board_t> boards(count/8,uninit);
    for (board_t& board : boards)
      board = random_board(random);

    // Run a bunch of random table operations
    for (int step=0;step<count;step++) {
      // Pick a random board
      const int b = choice(random,boards.size());
      const symmetry_t symmetry = random_symmetry(random);
      const board_t board = transform_board(symmetry,boards[b]);
      const side_t side0 = unpack(board,0), side1 = unpack(board,1);
      const bool aggressive = b&1;
      superinfo_t info;
      info.known = random_super(random);
      info.wins = super_meaningless(board)&info.known;

      // If we're not at full depth, introduct some errors in favor of white 
      int depth = choice(random);
      if (!depth) {
        const super_t errors = super_meaningless(board,1831);
        info.wins = (aggressive?info.wins&~errors:info.wins|errors)&info.known;
      }

      // Verify that lookup returns consistent results
      total_lookups++;
      superlookup_t data = aggressive?super_lookup<true >(depth,side0,side1)
                                     :super_lookup<false>(depth,side0,side1);
      if (data.info.known) {
        GEODE_ASSERT(info.known&data.info.known); // should happen with overwhelming probability
        successful_lookups++;
      }
      super_t diff = (info.wins^data.info.wins)&info.known&data.info.known;
      if (!depth)
        diff &= aggressive?info.wins:~info.wins;
      GEODE_ASSERT(!diff);

      // Optionally store
      if (choice(random)) {
        data.info = info;
        aggressive?super_store<true >(depth,data)
                  :super_store<false>(depth,data);
      }
    }
    GEODE_ASSERT(successful_lookups>count/16);
  }
  slog("total lookups = %d", total_lookups);
}

TEST(search, supertable) {
  supertable_test(10);
}

}
}
