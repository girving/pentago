// Generate the stored positions and exact values for cascade_test.cc
//
// usage: bazel run -c opt //pentago/cascade:gen_cascade_cases 2>/dev/null | grep -v "^(peak\|expanded nodes"
//
// For each slice 18-26, finds random and sensible-play boards settled quickly by each cascade stage
// (superengine, tie prover, midsolve), and prints {raw board, exact value, stage}.
#include "pentago/cascade/cascade.h"
#include "pentago/base/score.h"
#include "pentago/utility/random.h"
#include "pentago/utility/wall_time.h"
#include <cstdio>
#include <vector>
using namespace pentago;
using std::vector;

static high_board_t sensible_position(Random& random, const int stones) {
  for (;;) {
    side_t side[2] = {0, 0};
    int n = 0;
    for (; n < stones; n++) {
      const int t = n & 1;
      bool moved = false;
      for (int attempt = 0; attempt < 100 && !moved; attempt++) {
        const int c = random.uniform<int>(0, 36);
        const side_t x = side_t(1) << (16*(c/9) + c%9);
        if (x & (side[0] | side[1])) continue;
        const int q = random.uniform<int>(0, 4), d = random.uniform<int>(0, 2) ? 1 : -1;
        const auto after = high_board_t(t ? side[0] : side[0] | x, t ? side[1] | x : side[1], 2*n + 1).rotate(q, d);
        if (won(side[t] | x) || won(after.side(0)) || won(after.side(1))) continue;
        side[0] = after.side(0), side[1] = after.side(1), moved = true;
      }
      if (!moved) break;
    }
    if (n == stones) return high_board_t(side[0], side[1], 2*stones);
  }
}

int main() {
  Random random(27182);
  cascade_t cascade;
  printf("const cascade_case_t cascade_cases[] = {\n");
  for (int stones = 18; stones <= 26; stones++) {
    int found[3] = {0, 0, 0};
    const int want[3] = {2, 2, stones >= 21 ? 1 : 0};
    for (int attempt = 0; attempt < 400 && (found[0] < want[0] || found[1] < want[1] || found[2] < want[2]); attempt++) {
      high_board_t board = attempt & 1 ? sensible_position(random, stones)
                                       : high_board_t::from_board(random_board(random, stones), false);
      if (board.done()) continue;
      const auto start = wall_time();
      const auto r = cascade.value(board);
      const double t = (wall_time() - start).seconds();
      const int s = int(r.stage);
      if (found[s] >= want[s] || t > (s == 2 ? 2.0 : 0.5)) continue;
      found[s]++;
      const int v = cascade.midsolve(board);
      GEODE_ASSERT(v == r.value);
      printf("  {%lluull, %d, %d},  // %d stones, %.3f s\n", (unsigned long long)board.raw(), v, s, stones, t);
    }
    fprintf(stderr, "stones %d: found %d %d %d\n", stones, found[0], found[1], found[2]);
  }
  printf("};\n");
  return 0;
}
