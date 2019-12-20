#include "pentago/base/board.h"
#include "pentago/mid/midengine.h"
#include "pentago/search/superengine.h"
#include "pentago/search/supertable.h"
#include "pentago/utility/arange.h"
#include "pentago/utility/log.h"
#include "gtest/gtest.h"
#include <unordered_set>

namespace pentago {
namespace {

using std::get;
using std::unordered_set;

board_t random_board_at_slice(Random& random, const int stones) {
  GEODE_ASSERT(0<=stones && stones<=36);
  Array<int> flat(36);
  const auto entries = arange(36);
  random.shuffle(entries);
  for (const int i : range(stones/2)) flat[i] = 2;
  for (const int i : range(stones/2, stones)) flat[i] = 1;
  return from_table(flat.reshape(vec(6, 6)));
}

void midsolve_internal_test(const board_t board, const bool parity) {
  const int slice = count_stones(board);
  const auto workspace = midsolve_workspace(slice);
  const auto results = midsolve_internal(board,parity,workspace);
  ASSERT_EQ(results.size(), 37-slice); // Only mostly true due to superstandardization, but still good
  for (const auto& r : results) {
    const auto rboard = get<0>(r);
    const bool turn = count_stones(get<0>(r))&1;
    const auto rs = get<1>(r);
    ASSERT_EQ(popcount(rs.known), 128);
    slog("slice %d, board %19lld, parity %d: win %3d, tie %3d, loss %3d",
         slice, rboard, parity, popcount(rs.win), popcount(~rs.win&rs.notlose),
         popcount(~(rs.win|rs.notlose)));
    for (int a=0;a<2;a++) {
      super_t correct = super_evaluate_all(a,100,flip_board(rboard,turn));
      GEODE_ASSERT(!((correct^(a?rs.win:rs.notlose))&rs.known));
    }
  }
}

TEST(mid, internal) {
  Random random(5554);
  init_supertable(20);
  for (int slice = 36; slice >= 30; slice--) {
    for (int i = 0; i < 16; i++) {
      const auto board = random_board_at_slice(random, slice);
      for (const bool parity : {false, true})
        midsolve_internal_test(board, parity);
    }
  }
}

TEST(mid, mid) {
  Random random(2223);
  init_supertable(20);
  const auto workspace = midsolve_workspace(30);
  const auto empty = empty_block_cache();
  for (const int slice : range(30, 35+1)) {
    for (int i = 0; i < 16; i++) {
      const auto root = random_board_at_slice(random, slice);
      for (const bool middle : {false, true}) {
        const high_board_t high(root, middle);
        unordered_set<high_board_t> moves;
        if (middle)
          for (const auto& m : high.moves())
            moves.insert(m);
        else
          for (const auto& a : high.moves())
            for (const auto& b : a.moves())
              moves.insert(b);
        vector<board_t> boards;
        for (const auto& m : moves)
          boards.push_back(m.board);
        const auto values = midsolve(root, middle, boards, workspace);
        ASSERT_EQ(values.size(), moves.size());
        for (const auto& m : moves)
          ASSERT_EQ(check_get(values, m.board), m.value(*empty));
      }
    }
  }
}

halfsuper_t slow_split(const super_t s, const bool parity) {
  halfsuper_t h = 0;
  for (int a=0;a<2;a++)
    for (int b=0;b<4;b++)
      for (int c=0;c<4;c++)
        for (int d=0;d<4;d++)
          if (s(2*a+((b+c+d+parity)&1),b,c,d))
            h |= halfsuper_t::singleton(a,b,c,d);
  return h;
}

// For testing purposes
__attribute__((unused)) static super_t slow_merge(const halfsuper_t even, const halfsuper_t odd) {
  super_t s = 0;
  for (int a=0;a<2;a++)
    for (int b=0;b<4;b++)
      for (int c=0;c<4;c++)
        for (int d=0;d<4;d++) {
          if (even(a,b,c,d)) s |= super_t::singleton(2*a+((b+c+d  )&1),b,c,d);
          if ( odd(a,b,c,d)) s |= super_t::singleton(2*a+((b+c+d+1)&1),b,c,d);
        }
  return s;
}

superinfo_t info(const halfsuper_t h, const bool parity) {
  superinfo_t i;
  i.known = parity ? merge(0,~halfsuper_t(0)) : merge(~halfsuper_t(0),0);
  i.wins = parity ? merge(0,h) : merge(h,0);
  return i;
}

super_t random_super_and(Random& random, const int steps) {
  super_t s = random_super(random);
  for (int i=1;i<steps;i++)
    s &= random_super(random);
  return s;
}

TEST(mid, half) {
  const int steps = 1024;
  const bool verbose = false;
  Random random(667731);

  // Test split and merge
  for (int step=0;step<steps;step++) {
    const super_t s = step<256 ? super_t::singleton(step)
                               : random_super_and(random,4);
    const auto h = split(s);
    if (verbose)
      slog("---------------------\ns\n%g\nh0\n%s\nh1\n%s", s, info(h[0],0), info(h[1],1));

    // Test split and merge
    if (verbose)
      slog("slow h0\n%s\nslow h1\n%s\nmerge(h0,h1)\n%s",
           info(slow_split(s,0),0), info(slow_split(s,1),1), merge(h[0], h[1]));
    ASSERT_EQ(slow_split(s, 0), h[0]);
    ASSERT_EQ(slow_split(s, 1), h[1]);
    ASSERT_EQ(s, merge(h[0], h[1]));

    // Test rmax.  The order is flipped since rmax reversed parity.
    if (verbose) {
      slog("rmax(s) = %d\n%s", popcount(rmax(s)), rmax(s));
      slog("rmax(h0) = %d\n%s", popcount(rmax(h[0])), info(rmax(h[0]),1));
      slog("rmax(h1) = %d\n%s", popcount(rmax(h[1])), info(rmax(h[1]),0));
    }
    ASSERT_EQ(merge(rmax(h[1]), rmax(h[0])), rmax(s));
  }

  // Test wins
  for (int step=0;step<steps;step++) {
    const side_t side = random_side(random);
    const auto h = halfsuper_wins(side);
    ASSERT_EQ(super_wins(side), merge(h[0], h[1]));
  }
}

// Show that halfsuper_t is the best we can do: the parity
// configuration is the limit of rmax applied to a singleton.
TEST(mid, rmax_limit) {
  // Compute limit configuration
  auto s = super_t::singleton(0);
  vector<super_t> seen;
  for (int n=0;;n++) {
    if (std::count(seen.begin(), seen.end(), s))
      break;
    if (0)
      slog("n = %d\n%s\n", n, s);
    seen.push_back(s);
    s = rmax(s);
  }

  // Verify against parity
  for (const int i0 : range(4))
    for (const int i1 : range(4))
      for (const int i2 : range(4))
        for (const int i3 : range(4))
          ASSERT_EQ(s(i0, i1, i2, i3), (i0 + i1 + i2 + i3) & 1);
}

}  // namespace
}  // namespace pentago
