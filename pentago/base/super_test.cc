#include "pentago/base/count.h"
#include "pentago/base/score.h"
#include "pentago/base/symmetry.h"
#include "pentago/utility/integer_log.h"
#include "pentago/utility/range.h"
#include "pentago/utility/log.h"
#include "gtest/gtest.h"
#include <numeric>
#include <unordered_set>
namespace pentago {
namespace {

using std::abs;
using std::min;
using std::swap;
using std::unordered_set;

TEST(super, wins) {
  const int steps = 100000;

  // Determine the expected number of stones if we pick k stones with replacement
  Array<double> expected(32,uninit);
  for (int k=0;k<expected.size();k++)
    expected[k] = 36*(1-pow(1-1./36,k));
  
  // Test super_wins
  Random random(1740291);
  vector<int> counts(37);
  std::iota(counts.begin(), counts.end(), 0);
  swap(counts[0], counts[5]);
  for (int count : counts) {
    // Determine how many stones we need to pick with replacement to get roughly count stones
    const bool flip = count > 18;
    const int flip_count = flip ? 36 - count : count;
    int k = 0;
    for (int i = 0; i < expected.size(); i++)
      if (abs(expected[i] - flip_count) < abs(expected[k] - flip_count))
        k = i;
    ASSERT_LT(k, 30);
    slog("count %d, k %d, expected %d", count, k, flip?36-expected[k]:expected[k]);
      
    for (int step=0;step<steps;step++) {
      // Generate a random side with roughly count stones
      side_t side = 0;
      for (int j=0;j<k;j++) {
        unsigned p = random.uniform<int>(0,36);
        side |= (side_t)1<<(16*(p%4)+p/4);
      }
      if (flip)
        side ^= side_mask;

      // Compare super_wins to won
      super_t wins = super_wins(side);
      quadrant_t rotated[4][4];
      for (int q=0;q<4;q++) {
        rotated[q][0] = quadrant(side,q);
        for (int i=0;i<3;i++)
          rotated[q][i+1] = rotations[rotated[q][i]][0];
      }
      for (int r0=0;r0<4;r0++) for (int r1=0;r1<4;r1++) for (int r2=0;r2<4;r2++) for (int r3=0;r3<4;r3++) {
        side_t rside = quadrants(rotated[0][r0],rotated[1][r1],rotated[2][r2],rotated[3][r3]);
        if (won(rside)!=wins(r0,r1,r2,r3))
          THROW(AssertionError,"side %lld, rside %lld, r %d %d %d %d, correct %d, incorrect %d",side,rside,r0,r1,r2,r3,won(rside),wins(r0,r1,r2,r3));
      }
    }
  }
}

TEST(super, rmax) {
  const int steps = 100000;
  Random random(1740291);
  for (int step=0;step<steps;step++) {
    // Generate a random super_t
    super_t s = random_super(random);

    // Compare rmax with manual version
    super_t rs = rmax(s);
    for (int r0=0;r0<4;r0++) for (int r1=0;r1<4;r1++) for (int r2=0;r2<4;r2++) for (int r3=0;r3<4;r3++) {
      bool o = false;
      for (auto r : single_rotations)
        o |= s(vec(r0,r1,r2,r3)+r);
      ASSERT_EQ(rs(r0,r1,r2,r3), o);
    }
  }
}

TEST(super, bool) {
  ASSERT_FALSE(super_t(0));
  for (int i0=0;i0<4;i0++) for (int i1=0;i1<4;i1++) for (int i2=0;i2<4;i2++) for (int i3=0;i3<4;i3++) {
    ASSERT_TRUE(super_t::singleton(i0,i1,i2,i3));
    for (int j0=0;j0<4;j0++) for (int j1=0;j1<4;j1++) for (int j2=0;j2<4;j2++) for (int j3=0;j3<4;j3++)
      ASSERT_TRUE(super_t::singleton(i0,i1,i2,i3)|super_t::singleton(j0,j1,j2,j3));
  }
}

TEST(super, group) {
  // Test identity and inverses
  const symmetry_t e = 0;
  for (auto g : symmetries) {
    ASSERT_EQ(g * e, g);
    ASSERT_EQ(e * g, g);
    ASSERT_EQ(g * g.inverse(), e);
  }

  // Test cancellativitiy
  for (auto a : symmetries)
    for (auto b : symmetries) {
      ASSERT_EQ((b * a.inverse()) * a, b);
      ASSERT_EQ(a * (a.inverse() * b), b);
    }

  // Test associativity following Rajagopalany and Schulman, Verification of identities, 1999.
  // First, find a generating subset.
  vector<symmetry_t> generators;
  vector<symmetry_t> generated;
  generated.reserve(symmetries.size());
  unordered_set<symmetry_t> generated_set;
  generated.push_back(e);
  generated_set.insert(e);
  for (auto g : symmetries)
    if (generated_set.find(g) == generated_set.end()) {
      generators.push_back(g);
      for (;;) {
        RawArray<const symmetry_t> previous = generated;
        for (auto a : previous)
          if (generated_set.insert(g*a).second)
            generated.push_back(g*a);
        if (previous.size() == int(generated.size()))
          break;
      }
    }
  ASSERT_EQ(generated.size(), symmetries.size());
  ASSERT_LE(generators.size(), integer_log(symmetries.size())+2);
  // Check associativity on the generators.  This is sufficient thanks to cancellativity.
  for (auto a : generators)
    for (auto b : generators)
      for (auto c : generators)
        ASSERT_EQ((a*b)*c, a*(b*c));

  // Check products of local with general
  for (auto a : symmetries)
    for (int r : range(256)) {
      local_symmetry_t b(r);
      symmetry_t sb(b);
      ASSERT_EQ(a*b, a*sb);
      ASSERT_EQ(b*a, sb*a);
    }
}

TEST(super, action) {
  const int steps = 100000;
  Random random(875431);
  for (int step=0;step<steps;step++) {
    // Generate two random symmetries and sides
    const symmetry_t s0 = random_symmetry(random), s1 = random_symmetry(random);
    const side_t side0 = random_side(random), side1 = random_side(random)&~side0;
    // Check action consistency
    ASSERT_EQ(transform_side(s0,transform_side(s1,side0)), transform_side(s0*s1,side0));
    ASSERT_EQ(transform_board(s0,pack(side0,side1)),
              pack(transform_side(s0,side0),transform_side(s0,side1)));
  }
}

TEST(super, superstandardize) {
  const int steps = 10000;
  Random random(875431);
  for (int step=0;step<steps;step++) {
    // Generate a random board, with possibly duplicated quadrants
    const side_t side0 = random_side(random),
                 side1 = random_side(random)&~side0;
    const board_t pre = pack(side0,side1);
    const int q = random.uniform<int>(0,256);
    const board_t board = transform_board(random_symmetry(random),quadrants(quadrant(pre,q&3),quadrant(pre,q>>2&3),quadrant(pre,q>>4&3),quadrant(pre,q>>6&3)));
    // Standardize
    const auto [standard, symmetry] = superstandardize(board);
    ASSERT_EQ(transform_board(symmetry,board), standard);
    // Compare with manual version
    board_t slow = (uint64_t)-1;
    for (auto s : symmetries)
      slow = min(slow,transform_board(s,board));
    ASSERT_EQ(standard, slow);
  }
}

TEST(super, super_action) {
  const int steps = 10000;
  // First, test that transform_super satisfies our group theoretic definition:
  //   s(C) = gy(C) = {x in L | g'xgy in C} = {gzy'g' | z in C}
  Random random(72831);
  for (int step=0;step<steps;step++) {
    const symmetry_t s = random_symmetry(random);
    const symmetry_t sg(s.global,0), sl(0,s.local);
    const super_t C = random_super(random);
    super_t correct = 0;
    for (int r=0;r<256;r++)
      if (C(r))
        correct |= super_t::singleton((sg*symmetry_t(0,r)*(sl.inverse()*sg.inverse())).local);
    ASSERT_EQ(correct, transform_super(s,C));
  }

  // Next, test the motivating definition in terms of invariant functions
  for (int step=0;step<steps;step++) {
    const symmetry_t s = random_symmetry(random);
    const side_t side0 = random_side(random),
                 side1 = random_side(random)&~side0;
    const board_t board = pack(side0,side1);
    ASSERT_EQ(super_meaningless(transform_board(s, board)),
              transform_super(s, super_meaningless(board)));
  }
}

TEST(super, count) {
  const vector<int> counts = {1,3,30,227,2013,13065,90641,493844,2746022,12420352,56322888};
  for (const int n : range(counts.size())) {
    const int count = counts[n];
    slog("n %d, correct %d, computed %d", n, count, count_boards(n, 2048));
    ASSERT_EQ(count_boards(n, 2048), count);
  }
}

TEST(super, meaningless) {
  // Regression test for meaningless
  const bool regenerate = false;
  if (regenerate) {
    Random random(17);
    string s = "static const tuple<board_t,bool> golden[] = {";
    for (const int i : range(20)) {
      const auto board = random_board(random);
      s += format("{%d,%d}%s", board, meaningless(board), i<19 ? "," : "};");
    }
    slog(s);
  } else {
    // Generated by the branch above
    static const tuple<board_t,bool> golden[] = {{798570845956604183,1},{229736843749230197,1},{185776620857593483,0},{3512837052653386215,0},{684286774138772602,0},{5198850126605321652,1},{343722965088534634,0},{4904424083383321143,1},{3746797175594829623,1},{2486550524810956973,0},{1233781166207355024,0},{3878447699434279408,0},{3834844314683386553,0},{4119114683469660182,1},{3262014742351909660,1},{2293807540346952111,1},{5413048756049160647,0},{3703696599961057535,1},{4747358366095061118,0},{2321954716496433629,1}};
    for (const auto& [board, m] : golden)
      ASSERT_EQ(m, meaningless(board));
  }
}

}  // namespace
}  // namespace pentago
