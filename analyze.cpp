// Various analysis code

#include <pentago/section.h>
#include <pentago/superscore.h>
#include <pentago/utility/wall_time.h>
#include <other/core/array/Array2d.h>
#include <other/core/geometry/Box.h>
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

// Rasterize a piece of history data into an rgba image
void rasterize_history(RawArray<Vector<real,4>,2> image, const Vector<real,2> sizes, RawArray<const Vector<wall_time_t,2>> history, const Box<real> y_range, const Vector<real,4> color) {
  typedef Vector<real,2> TV;
  typedef Vector<int,2> IV;
  auto scales = TV(image.sizes()+1)/sizes;
  scales.x *= 1e-6;
  const auto scaled_y_range = scales.y*y_range;
  for (const auto event : history) {
    const Box<TV> box(vec(scales.x*event.x.us,scaled_y_range.min),vec(scales.x*event.y.us,scaled_y_range.max));
    const Box<IV> ibox(clamp_min(IV(floor(box.min)),0),clamp_max(IV(ceil(box.max)),image.sizes()));
    for (int i : range(ibox.min.x,ibox.max.x))
      for (int j : range(ibox.min.y,ibox.max.y))
        image(i,j) += color*Box<TV>::intersect(box,Box<TV>(TV(i,j),TV(i+1,j+1))).volume();
  }
}

}
using namespace pentago;

void wrap_analyze() {
  OTHER_FUNCTION(sample_immediate_endings)
  OTHER_FUNCTION(simplify_history)
  OTHER_FUNCTION(rasterize_history)
}
