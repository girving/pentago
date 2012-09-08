// Various analysis code

#include <pentago/section.h>
#include <pentago/superscore.h>
#include <pentago/utility/wall_time.h>
#include <other/core/array/Array.h>
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

// Merge consecutive time intervals separated by at most a threshold
Array<Vector<wall_time_t,2>> simplify_history(RawArray<const Vector<wall_time_t,2>> history, int threshold) {
  Array<Vector<wall_time_t,2>> merged;
  for (int i=0;i<history.size();i++) {
    wall_time_t start,end;
    history[i].get(start,end);
    while (history.valid(i+1) && (history[i+1].x-end).us<=threshold)
      end = history[++i].y;
    merged.append(vec(start,end));
  }
  return merged;
}

}
using namespace pentago;

void wrap_analyze() {
  OTHER_FUNCTION(sample_immediate_endings)
  OTHER_FUNCTION(simplify_history)
}
