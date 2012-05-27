// Various analysis code

#include "all_boards.h"
#include "superscore.h"
#include <other/core/python/module.h>
#include <other/core/random/Random.h>
#include <other/core/utility/range.h>
namespace pentago {

// Generate n random boards, and count the number of (boring positions, immediate black wins, immediate wins white, and ties)
Vector<int,4> sample_immediate_endings(Random& random, section_t section, int samples) {
  Vector<int,4> counts;
  for (int i=0;i<samples;i++) {
    const board_t board = random_board(random,section);
    const side_t side0 = unpack(board,0),
                 side1 = unpack(board,1);
    const super_t wins0 = super_wins(side0),
                  wins1 = super_wins(side1);
    counts[1] += popcount(wins0&~wins1);
    counts[2] += popcount(~wins0&wins1);
    counts[3] += popcount(wins0&wins1);
  }
  counts[0] = 256*samples-counts.sum();
  return counts;
}

}
using namespace pentago;

void wrap_analyze() {
  OTHER_FUNCTION(sample_immediate_endings)
}
