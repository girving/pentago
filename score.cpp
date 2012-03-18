// Score definitions and immediately evaluation functions

#include "score.h"
#include <other/core/math/min.h>
#include <other/core/math/popcount.h>
#include <other/core/python/module.h>
namespace pentago {

using namespace other;

// Compute the minimum number of black moves required for a win, together with the number of different
// ways that minimum can be achieved.  Returns ((6-min_distance)<<16)+count, so that a higher number means
// closer to a black win.  If winning is impossible, the return value is 0.
int rotated_win_closeness(side_t black, side_t white) {
  // Transpose into packed quadrants
  const quadrant_t q0 = pack(quadrant(black,0),quadrant(white,0)),
                   q1 = pack(quadrant(black,1),quadrant(white,1)),
                   q2 = pack(quadrant(black,2),quadrant(white,2)),
                   q3 = pack(quadrant(black,3),quadrant(white,3));
  // Compute all distances
  const uint64_t each = 321685687669321; // sum_{i<17} 1<<3*i
  #define DISTANCES(i) ({ \
    const uint64_t d0 = rotated_win_distances[0][q0][i], \
                   d1 = rotated_win_distances[1][q1][i], \
                   d2 = rotated_win_distances[2][q2][i], \
                   d3 = rotated_win_distances[3][q3][i], \
                   blocks = (d0|d1|d2|d3)&4*each, \
                   blocked = blocks|blocks>>1|blocks>>2, \
                   unblocked = ~blocked; \
    (d0&unblocked)+(d1&unblocked)+(d2&unblocked)+(d3&unblocked)+(6*each&blocked); })
  const uint64_t d0 = DISTANCES(0), // Each of these contains 17 3-bit distances in [0,6]
                 d1 = DISTANCES(1),
                 d2 = DISTANCES(2),
                 d3 = DISTANCES(3);
  #undef DISTANCES
  // Determine minimum distance
  const bool min_under_1 = (~((d0|d0>>1|d0>>2)&(d1|d1>>1|d1>>2)&(d2|d2>>1|d2>>2)&(d3|d3>>1|d3>>2))&each)!=0, // abc < 1 iff ~(a|b|c)
             min_under_2 = (~((d0|d0>>1)&(d1|d1>>1)&(d2|d2>>1)&(d3|d3>>1))&2*each)!=0, // abc < 2 iff ~a&~b = ~(a|b)
             min_under_3 = (~((d0>>2|(d0>>1&d0))&(d1>>2|(d1>>1&d1))&(d2>>2|(d2>>1&d2))&(d3>>2|(d3>>1&d3)))&each)!=0, // abc < 3 iff ~a&(~b|~c) = ~a&~(b&c) = ~(a|(b&c))
             min_under_4 = (~(d0&d1&d2&d3)&4*each)!=0, // abc < 4 iff ~a
             min_under_5 = (~((d0>>2&(d0>>1|d0))&(d1>>2&(d1>>1|d1))&(d2>>2&(d2>>1|d2))&(d3>>2&(d3>>1|d3)))&each)!=0, // abc < 5 iff ~a|(~b&~c) = ~a|~(b|c) = ~(a&(b|c))
             min_under_6 = (~((d0&d0>>1)&(d1&d1>>1)&(d2&d2>>1)&(d3&d3>>1))&2*each)!=0; // abc < 6 iff ~a|~b = ~(a&b)
  const int min_distance = min_under_4
                             ?min_under_2
                               ?min_under_1?0:1
                               :min_under_3?2:3
                             :min_under_5
                               ?4
                               :min_under_6?5:6;
  // If we're in debug mode, check against the slow way
#ifndef NDEBUG
  #define SLOW_MIN_DISTANCE(d) ({ \
    int md = 6; \
    for (int i=0;i<17;i++) \
      md = min(md,int(d>>3*i)&7); \
    md; })
  const int slow_min_distance = min(SLOW_MIN_DISTANCE(d0),SLOW_MIN_DISTANCE(d1),SLOW_MIN_DISTANCE(d2),SLOW_MIN_DISTANCE(d3));
  OTHER_ASSERT(slow_min_distance==min_distance);
#endif
  // If the minimum distance is 6, a black win is impossible, so no need to count the ways
  if (min_distance==6)
    return 0;
  // Count number of times min_distance occurs
  const uint64_t mins = min_distance*each;
  #define MATCHES(d) (~((d^mins)|(d^mins)>>1|(d^mins)>>2)&each)
  const int count = popcount(MATCHES(d0))+popcount(MATCHES(d1)|MATCHES(d2)<<1|MATCHES(d3)<<2);
  #undef MATCHES
  return ((6-min_distance)<<16)+count;
}

static int rotated_win_closeness_py(board_t board) {
  check_board(board);
  return rotated_win_closeness(unpack(board,0),unpack(board,1));
}

static int status_py(board_t board) {
  check_board(board);
  return status(board);
}

static int rotated_status(board_t board) {
  check_board(board);
  return rotated_won(unpack(board,0))?1:0;
}


}
using namespace pentago;
using namespace other::python;

void wrap_score() {
  function("status",status_py);
  function("rotated_win_closeness",rotated_win_closeness_py);
  OTHER_FUNCTION(rotated_status)
}
