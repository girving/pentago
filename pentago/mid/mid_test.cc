#include "pentago/base/board.h"
#include "pentago/mid/midengine.h"
#include "pentago/mid/internal.h"
#include "pentago/mid/subsets.h"
#include "pentago/high/check.h"
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

// The boards corresponding to midsolve_internal's results
Array<const board_t> result_boards(const high_board_t board) {
  vector<board_t> boards = {board.board()};
  const auto empty = board.empty_mask();
  for (const int bit : range(64)) {
    if (empty & side_t(1)<<bit) {
      const auto move = side_t(1) << bit;
      side_t after[2] = {board.side(0), board.side(1)};
      after[board.count() & 1] |= move;
      boards.push_back(pack(after[0], after[1]));
    }
  }
  return asarray(boards).copy();
}

void midsolve_internal_test(const high_board_t board) {
  typedef halfsuper_t h;
  const int slice = board.count();
  const bool parity = board.middle();
  const auto workspace = midsolve_workspace(slice);
  const Vector<halfsupers_t,1+18> padded_results = midsolve_internal(board, workspace);
  const auto results = asarray(padded_results).slice(0, mid_supers_size(board));
  const auto boards = result_boards(board);
  ASSERT_EQ(results.size(), 37-slice);  // Only mostly true due to superstandardization, but still good
  ASSERT_EQ(results.size(), boards.size());
  for (const int i : range(results.size())) {
    const auto rs = results[i];
    const auto rboard = boards[i];
    const bool turn = count_stones(rboard) & 1;
    const bool rparity = (i > 0) != parity;
    const bool verbose = false;
    if (verbose)
      slog("slice %d, board %19lld, parity %d: win %3d, tie %3d, loss %3d",
           slice, rboard, parity, popcount(rs.win), popcount(~h(rs.win) & h(rs.notlose)),
           popcount(~(h(rs.win) | h(rs.notlose))));
    for (const int s : range(128)) {
      // Assert win implies notlose
      GEODE_ASSERT(pentago::get(rs.notlose, s) >= pentago::get(rs.win, s),
                   format("slice %d, board %d, s %d, notlose %d, win %d", s,
                          slice, rboard, pentago::get(rs.notlose, s), pentago::get(rs.win, s)));
    }
    for (const int a : range(2)) {
      const auto exp = [=](const halfsuper_t h) { return rparity ? merge(0,h) : merge(h,0); };
      const auto known = exp(~halfsuper_t(0));
      const super_t correct = super_evaluate_all(a, 100, flip_board(rboard, turn));
      for (const int s : range(256))
        if (known[s])
          ASSERT_EQ(correct[s], value(rs, s) >= a);
      GEODE_ASSERT(!((correct ^ exp(a ? rs.win : rs.notlose)) & known), format("slice %d, board %s, a %d", slice, rboard, a));
    }
  }
}

TEST(mid, internal) {
  Random random(5554);
  init_supertable(20, false);
  for (int slice = 36; slice >= 21; slice--) {
    for (int i = 0; i < 16; i++) {
      const auto board = random_board_at_slice(random, slice);
      for (const bool parity : {false, true})
        midsolve_internal_test(high_board_t::from_board(board, parity));
    }
  }
}

TEST(mid, mid) {
  Random random(2223);
  init_supertable(20, false);
  const auto workspace = midsolve_workspace(30);
  const auto empty = empty_block_cache();
  for (const int slice : range(30, 35+1)) {
    for (int i = 0; i < 8192; i++) {
      const auto root = random_board_at_slice(random, slice);
      for (const bool middle : {false, true}) {
        const auto high = high_board_t::from_board(root, middle);
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
          ASSERT_EQ(get<1>(*it), value(*empty, b));
        }
      }
    }
  }
}

// Regression test, since I'm paranoid and am about to rewrite the routine in question
TEST(mid, bottleneck) {
  const int correct[] = {31855824, 11435424, 4036032, 1387386, 504504, 180180, 62370,
                         23100, 8400, 2940, 1120, 420, 150, 60, 24, 9, 4, 2};
  for (const int slice : range(18, 36))
    ASSERT_EQ(midsolve_workspace(slice).size(), correct[slice - 18]);
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
  using pentago::get;
  for (const int n : range(18+1)) {
    for (const int k : range(4)) {
      const auto sets = make_sets(n, k);
      ASSERT_EQ(sets.size, choose(n, k));
      int a = 0;
      switch (k) {
        case 0:
          ASSERT_EQ(get(sets, a++), 0);
          break;
        case 1:
          for (const int i : range(n))
            ASSERT_EQ(get(sets, a++), i);
          break;
        case 2:
          for (const int i0 : range(n))
            for (const int i1 : range(n))
              if (i0 > i1)
                ASSERT_EQ(get(sets, a++), i1|i0<<5);
          break;
        case 3:
          for (const int i0 : range(n))
            for (const int i1 : range(n))
              for (const int i2 : range(n))
                if (i0 > i1 && i1 > i2)
                  ASSERT_EQ(get(sets, a++), i2|i1<<5|i0<<10);
          break;
      }
      ASSERT_EQ(sets.size, a);
    }
  }
}

TEST(mid, subset_indices) {
  using pentago::get;
  for (const int n : range(18+1)) {
    for (const int k : range(4)) {
      const auto sets = make_sets(n, k);
      for (const int s : range(sets.size)) {
        const auto set = get(sets, s);
        const auto mask = subset_mask(sets, s);
        ASSERT_EQ(s, subset_index(sets, set));
        ASSERT_EQ(k, popcount(mask));
        for (const int i : range(k)) {
          const int j = set>>5*i&0x1f;
          GEODE_ASSERT(mask & 1<<j);
        }
      }
    }
  }
}

TEST(mid, cs0ps) {
  for (const int n : range(18)) {
    for (const int k : range(4)) {
      const auto sets0p = make_sets(n, k);
      const auto csets0p = make_sets(n, k+1);
      for (const int s0p : range(sets0p.size)) {
        const auto mask = subset_mask(sets0p, s0p);
        int m = 0;
        for (const int i : range(n)) {
          if (!(mask & 1<<i)) {
            const auto cs0p = make_cs0ps(sets0p, s0p, m++);
            const auto cmask = mask | 1<<i;
            ASSERT_EQ(cmask, subset_mask(csets0p, cs0p))
              << format("n %d, k %d, s0p %d, mask %d, i %d, m %d, cs0p %d, cmask %d",
                        n, k, s0p, mask, i, m-1, cs0p, cmask);
          }
        }
      }
    }
  }
}

static inline uint32_t absolute_mask(const uint32_t taken, const uint32_t relative) {
  GEODE_ASSERT(popcount(taken) + popcount(relative) <= 18);
  uint8_t empty[18];
  for (int next = 0, i = 0; i < 18; i++)
    if (!(taken & 1<<i))
      empty[next++] = i;
  uint32_t absolute = 0;
  for (int i = 0; i < 18; i++)
    if (relative & 1<<i)
      absolute |= 1<<empty[i];
  return absolute;
}

static string bin(int n) {
  string s;
  do {
    s.push_back('0' + (n & 1));
    n >>= 1;
  } while (n);
  s += "b0";
  std::reverse(s.begin(), s.end());
  return s;
}

TEST(mid, set1_info) {
  Random random(7);
  for (int sample = 0; sample < (1<<17); sample++) {
    // Sample a random call to set1_info_t::commute
    const int slice = random.uniform<int>(18, 36+1);
    const auto board = high_board_t::from_board(random_board(random, slice), sample & 1);
    const auto I = make_transposed(board);
    const int n = random.uniform<int>(I.spots+1);
    const helper_t<transposed_t> H{I, n};
    const auto sets1 = H.sets1(), sets1p = H.sets1p();
    const int s1 = random.uniform<int>(sets1.size);
    const auto I1 = make_set1_info(I, n, s1);
    const bool next = !H.done() && random.uniform<int>(2);
    const auto sets0 = make_sets(I.spots, H.k0()+next);
    const auto sets0p = make_sets(I.spots-H.k1(), H.k0()+next);
    const int s0p = random.uniform<int>(sets0p.size);
    const auto c = commute(I1, sets0p, s0p);

    // Verify consistency
    GEODE_ASSERT(unsigned(c.s0) < unsigned(sets0.size));
    GEODE_ASSERT(unsigned(c.s1p) < unsigned(sets1p.size));
    const auto mask1 = subset_mask(sets1, s1);
    const auto mask0p = subset_mask(sets0p, s0p);
    const auto mask0 = subset_mask(sets0, c.s0);
    const auto mask1p = subset_mask(sets1p, c.s1p);
    const auto amask0 = absolute_mask(mask1, mask0p);
    const auto amask1 = absolute_mask(mask0, mask1p);
    if (amask0 != mask0 || amask1 != mask1) {
      slog("bad:");
      slog("  slice %d, n %d, next %d", slice, n, next);
      slog("  spots %d, k0 %d, k1 %d", I.spots, H.k0(), H.k1());
      slog("  s1 %d, s0p %d -> s0 %d, s1p %d", s1, s0p, c.s0, c.s1p);
      slog("  mask1 %s, mask0p %s -> mask0 %s, mask1p %s", bin(mask1), bin(mask0p), bin(mask0), bin(mask1p));
      slog("  amask0 %s, amask1 %s", bin(amask0), bin(amask1));
    }
    ASSERT_EQ(amask0, mask0);
    ASSERT_EQ(amask1, mask1);
  }
}

}  // namespace
}  // namespace pentago
