// Various analysis code

#include <pentago/base/score.h>
#include <pentago/base/section.h>
#include <pentago/base/superscore.h>
#include <pentago/base/symmetry.h>
#include <pentago/utility/ceil_div.h>
#include <pentago/utility/convert.h>
#include <pentago/utility/spinlock.h>
#include <pentago/utility/thread.h>
#include <pentago/utility/wall_time.h>
#include <geode/array/Array2d.h>
#include <geode/geometry/Box.h>
#include <geode/math/uint128.h>
#include <geode/python/wrap.h>
#include <geode/random/Random.h>
#include <geode/utility/openmp.h>
#include <geode/utility/range.h>
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

static Vector<side_t,8> side_rotations(const side_t side) {
  Vector<side_t,8> r;
  for (int q=0;q<4;q++) {
    const auto masked = side&~(side_t(0x1ff)<<16*q);
    const auto sq = quadrant(side,q);
    for (int d=0;d<2;d++)
      r[q*2+d] = masked|rotations[sq][d]<<16*q;
  }
  return r;
}

// popcount(side0) is the side who played last
static bool reachable(const side_t side0, const side_t side1) {
  if (!side0)
    return true;
  const auto r0 = side_rotations(side0),
             r1 = side_rotations(side1);
  for (int r=0;r<8;r++)
    if (!won(r0[r]) && !won(r1[r]))
      for (int q=0;q<4;q++)
        for (int x=0;x<3;x++)
          for (int y=0;y<3;y++) {
            const auto m = side_t(1)<<(16*q+3*x+y);
            if (r0[r]&m && reachable(side1,r0[r]&~m))
                return true;
          }
  return false;
}

static int orbit_size(const board_t board) {
  int count = 0;
  for (int g=0;g<8;g++)
    if (transform_board(symmetry_t(g,0),board)==board)
      count++;
  GEODE_ASSERT(8%count==0);
  return 8/count;
}

// Generate n random boards, counting the number of boards which don't have a path to the root
// within an intermediate win.  If orbit(board) is the size of the orbit under global rotations,
// we let
//   c = 8/orbit(board)
//   r = reachable(board)
// we return the sum and sum of squares of (c,cr,c(1-r))
typedef Tuple<Vector<uint64_t,3>,Vector<uint64_t,3>> reachable_t;
reachable_t sample_reachable_boards(const int slice, const int samples) {
  spinlock_t lock;
  reachable_t total;
  const int blocks = ceil_div(samples,100000);
  for (const int block : range(blocks)) {
    threads_schedule(CPU,[=,&total,&lock]() {
      const auto chunk = partition_loop(samples,blocks,block);
      const auto random = new_<Random>(chunk.lo);
      reachable_t sum;
      for (int s=0;s<chunk.size();s++) {
        const auto board = random_board(random,slice);
        const bool t = !(slice&1);
        const bool r = reachable(unpack(board,t),unpack(board,!t));
        const int c = 8/orbit_size(board);
        const auto x = Vector<uint64_t,3>(c,c*r,c*(1-r));
        sum.x += x;
        sum.y += sqr(x);
      }
      spin_t spin(lock);
      total.x += sum.x;
      total.y += sum.y;
    });
  }
  threads_wait_all();
  return total;
}

// Merge consecutive time intervals separated by at most a threshold
Array<Vector<wall_time_t,2>> simplify_history(RawArray<const history_t> history, int threshold) {
  Array<Vector<wall_time_t,2>> merged;
  for (int i=0;i<history.size();i++) {
    const wall_time_t start = history[i].start;
    wall_time_t end = history[i].end;
    while (history.valid(i+1) && (history[i+1].start-end).us<=threshold)
      end = history[++i].end;
    merged.append(vec(start,end));
  }
  return merged;
}

// Rasterize a piece of history data into an rgba image
void rasterize_history(RawArray<Vector<real,4>,2> image, const Box<Vector<real,2>> box,
                       RawArray<const history_t> history, const Box<real> y_range, const Vector<real,4> color) {
  typedef Vector<real,2> TV;
  typedef Vector<int,2> IV;
  const auto prescales = TV(image.sizes())/box.sizes();
  const auto scales = prescales*TV(1e-6,1);
  const auto scaled_y_range = scales.y*y_range;
  const auto base = prescales*box.min;
  for (const auto event : history) {
    const auto ebox = Box<TV>(vec(scales.x*event.start.us,scaled_y_range.min),
                              vec(scales.x*event.end.us  ,scaled_y_range.max))-Box<TV>(base);
    const Box<IV> ibox(clamp_min(IV(floor(ebox.min)),0),
                       clamp_max(IV( ceil(ebox.max)),image.sizes()));
    if (ibox.sizes().min() > 0)
      for (const int i : range(ibox.min.x,ibox.max.x))
        for (const int j : range(ibox.min.y,ibox.max.y))
          image(i,j) += color*Box<TV>::intersect(ebox,Box<TV>(TV(i,j),TV(i+1,j+1))).volume();
  }
}

// A benchmark for threefish random numbers
uint128_t threefry_benchmark(int n) {
  GEODE_ASSERT(n>=0);
  uint128_t result = n;
  for (int i=0;i<n;i++)
    result = threefry(result,i);
  return result;
}

}
using namespace pentago;

void wrap_analyze() {
  GEODE_FUNCTION(sample_immediate_endings)
  GEODE_FUNCTION(simplify_history)
  GEODE_FUNCTION(rasterize_history)
  GEODE_FUNCTION(threefry_benchmark)
  GEODE_FUNCTION(sample_reachable_boards)
}
