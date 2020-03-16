#include "pentago/base/board.h"
#include "pentago/base/count.h"
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
    slog("slice %d, board %19lld, parity %d: win %3d, tie %3d, loss %3d",
         slice, rboard, parity, popcount(rs.win), popcount(~rs.win&rs.notlose),
         popcount(~(rs.win|rs.notlose)));
    for (const int a : range(2)) {
      const auto exp = [=](const halfsuper_t h) { return rs.parity ? merge(0,h) : merge(h,0); };
      const auto known = exp(~halfsuper_t(0));
      const auto correct = super_evaluate_all(a,100,flip_board(rboard,turn));
      GEODE_ASSERT(!((correct ^ exp(a ? rs.win : rs.notlose)) & known));
      for (const int s : range(256)) {
        ASSERT_EQ(known[s], rs.known(s));
        if (rs.known(s))
          ASSERT_EQ(correct[s], rs.value(s) >= a);
      }
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
    for (int i = 0; i < 8192; i++) {
      const auto root = random_board_at_slice(random, slice);
      for (const bool middle : {false, true}) {
        const high_board_t high(root, middle);
        unordered_set<high_board_t> boards = {high};
        if (!high.done()) {
          if (middle)
            for (const auto& m : high.moves())
              boards.insert(m);
          else
            for (const auto& a : high.moves())
              for (const auto& b : a.moves())
                boards.insert(b);
        }
        const auto values = midsolve(high, workspace);
        ASSERT_EQ(values.size(), boards.size());
        for (const auto& b : boards) {
          const auto it = std::find_if(values.begin(), values.end(),
                                       [=](const auto& x) { return get<0>(x) == b; });
          ASSERT_NE(it, values.end());
          ASSERT_EQ(get<1>(*it), b.value(*empty));
        }
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

TEST(mid, half_split_merge) {
  const int steps = 1024;
  const bool verbose = false;
  Random random(667731);
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
}

TEST(mid, half_wins) {
  const int steps = 1 << 20;
  Random random(667731);
  const auto check = [](const side_t side) {
    const auto h0 = halfsuper_wins(side, 0);
    const auto h1 = halfsuper_wins(side, 1);
    return super_wins(side) == merge(h0, h1);
  };
  for (int step=0;step<steps;step++) {
    auto side = random_side(random);
    if (check(side)) continue;

    // Make minimal failing example
    for (;;) {
      for (const int i : range(64)) {
        const auto smaller = side & ~(side_t(1)<<i);
        if (smaller != side && !check(smaller)) {
          side = smaller;
          goto shrunk;
        }
      }
      break;
      shrunk:;
    }

    // Complain about it
    const auto h0 = halfsuper_wins(side, 0);
    const auto h1 = halfsuper_wins(side, 1);
    const auto wins = merge(h0, h1);
    const auto correct = super_wins(side);
    if (wins != correct) {
      for (const int i : range(256)) {
        if (wins[i] != correct[i]) {
          slog("i %d, r %d %d %d %d, wins %d, correct %d",
               i, i&3, i>>2&3, i>>4&3, i>>6&3, wins[i], correct[i]);
          const auto board = pack(side, 0);
          slog("side %d, board %d", side, board);
          slog("%s", str_board(board));
          break;
        }
      }
    }

    // Bail
    ASSERT_EQ(wins, correct);
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

TEST(mid, subsets) {
  for (const int n : range(18+1)) {
    for (const int k : range(4)) {
      const Array<set_t> sets(choose(n, k));
      subsets(n, k, sets);
      int a = 0;
      switch (k) {
        case 0:
          ASSERT_EQ(sets[a++], 0);
          break;
        case 1:
          for (const int i : range(n))
            ASSERT_EQ(sets[a++], i);
          break;
        case 2:
          for (const int i0 : range(n))
            for (const int i1 : range(n))
              if (i0 > i1)
                ASSERT_EQ(sets[a++], i1|i0<<5);
          break;
        case 3:
          for (const int i0 : range(n))
            for (const int i1 : range(n))
              for (const int i2 : range(n))
                if (i0 > i1 && i1 > i2)
                  ASSERT_EQ(sets[a++], i2|i1<<5|i0<<10);
          break;
      }
      ASSERT_EQ(sets.size(), a);
    }
  }
}

}  // namespace
}  // namespace pentago
