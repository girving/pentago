// Various analysis code

#include "pentago/base/score.h"
#include "pentago/base/section.h"
#include "pentago/base/superscore.h"
#include "pentago/base/symmetry.h"
#include "pentago/utility/ceil_div.h"
#include "pentago/utility/spinlock.h"
#include "pentago/utility/thread.h"
#include "pentago/utility/wall_time.h"
#include "pentago/utility/array.h"
#include "pentago/utility/box.h"
#include "pentago/utility/uint128.h"
#include "pentago/utility/random.h"
#include "pentago/utility/range.h"
#include "pentago/utility/sqr.h"
namespace pentago {

using std::get;
using std::max;
using std::min;

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
typedef tuple<Vector<uint64_t,3>,Vector<uint64_t,3>> reachable_t;
reachable_t sample_reachable_boards(const int slice, const int samples) {
  spinlock_t lock;
  reachable_t total;
  const int blocks = ceil_div(samples,100000);
  for (const int block : range(blocks)) {
    threads_schedule(CPU,[=,&total,&lock]() {
      const auto chunk = partition_loop(samples,blocks,block);
      Random random(chunk.lo);
      reachable_t sum;
      for (int s=0;s<chunk.size();s++) {
        const auto board = random_board(random,slice);
        const bool t = !(slice&1);
        const bool r = reachable(unpack(board,t),unpack(board,!t));
        const int c = 8/orbit_size(board);
        const auto x = Vector<uint64_t,3>(c,c*r,c*(1-r));
        get<0>(sum) += x;
        get<1>(sum) += sqr(x);
      }
      spin_t spin(lock);
      get<0>(total) += get<0>(sum);
      get<1>(total) += get<1>(sum);
    });
  }
  threads_wait_all();
  return total;
}

// Merge consecutive time intervals separated by at most a threshold
Array<Vector<wall_time_t,2>> simplify_history(RawArray<const history_t> history, int threshold) {
  vector<Vector<wall_time_t,2>> merged;
  for (int i=0;i<history.size();i++) {
    const wall_time_t start = history[i].start;
    wall_time_t end = history[i].end;
    while (history.valid(i+1) && (history[i+1].start-end).us<=threshold)
      end = history[++i].end;
    merged.push_back(vec(start,end));
  }
  return asarray(merged).copy();
}

// Rasterize a piece of history data into an rgba image
void rasterize_history(RawArray<Vector<float,4>,2> image, const Box<Vector<float,2>> box,
                       RawArray<const history_t> history, const Box<float> y_range, const Vector<float,4> color) {
  typedef Vector<float,2> TV;
  typedef Vector<int,2> IV;
  const auto prescales = TV(image.shape()) / box.shape();
  const auto scales = prescales*TV(1e-6,1);
  const auto scaled_ymin = scales[1]*y_range.min, scaled_ymax = scales[1]*y_range.max;
  const auto base = prescales*box.min;
  for (const auto event : history) {
    const auto emin = vec(scales[0]*event.start.us,scaled_ymin) - base;
    const auto emax = vec(scales[0]*event.end.us  ,scaled_ymax) - base;
    IV imin, imax;
    for (const int k : range(2)) {
      imin[k] = max(int(floor(emin[k])), 0);
      imax[k] = min(int( ceil(emax[k])), image.shape()[k]);
    }
    if ((imax - imin).min() > 0)
      for (const int i : range(imin[0],imax[0]))
        for (const int j : range(imin[1],imax[1]))
          image(i,j) += color * (min(emax[0],float(i+1))-max(emin[0],float(i))) *
                                (min(emax[1],float(j+1))-max(emin[1],float(j)));
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
