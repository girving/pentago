// Score definitions and immediately evaluation functions

#include "pentago/base/score.h"
#include "pentago/utility/debug.h"
#include "pentago/utility/popcount.h"
namespace pentago {

using std::min;

#define DISTANCES(table,i) ({ \
  const uint64_t d0 = table[0][q0][i], \
                 d1 = table[1][q1][i], \
                 d2 = table[2][q2][i], \
                 d3 = table[3][q3][i], \
                 blocks = (d0|d1|d2|d3)&4*each, \
                 blocked = blocks|blocks>>1|blocks>>2, \
                 unblocked = ~blocked; \
    (d0&unblocked)+(d1&unblocked)+(d2&unblocked)+(d3&unblocked)+(6*each&blocked); })

#define ROTATED_DISTANCES(i) DISTANCES(rotated_win_distances,i)
#define UNROTATED_DISTANCES(i) DISTANCES(unrotated_win_distances,i)
#define ARBITRARILY_ROTATED_DISTANCES(i) DISTANCES(arbitrarily_rotated_win_distances,i)

#define SLOW_MIN_DISTANCE(d,fields,width) ({ \
  int md = 6; \
  for (int i=0;i<fields;i++) \
    md = min(md,int(d>>width*i)&7); \
  md; })

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
  const uint64_t d0 = ROTATED_DISTANCES(0), // Each of these contains 17 3-bit distances in [0,6]
                 d1 = ROTATED_DISTANCES(1),
                 d2 = ROTATED_DISTANCES(2),
                 d3 = ROTATED_DISTANCES(3);
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
  const int slow_min_distance = min(min(SLOW_MIN_DISTANCE(d0,17,3),SLOW_MIN_DISTANCE(d1,17,3)),
                                    min(SLOW_MIN_DISTANCE(d2,17,3),SLOW_MIN_DISTANCE(d3,17,3)));
  GEODE_ASSERT(slow_min_distance==min_distance);
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

// Same as above, except using unrotated_win_distances or arbitrarily_rotated instead of rotated_win_distances (and therefore using 2 subfields instead of 4)
template<bool rotated> static int win_closeness(side_t black, side_t white) {
  // Transpose into packed quadrants
  const quadrant_t q0 = pack(quadrant(black,0),quadrant(white,0)),
                   q1 = pack(quadrant(black,1),quadrant(white,1)),
                   q2 = pack(quadrant(black,2),quadrant(white,2)),
                   q3 = pack(quadrant(black,3),quadrant(white,3));
  // Compute all distances
  const uint64_t each = 0x1111111111111111; // sum_{i<16} 1<<4*i
  const uint64_t d0 = rotated?ARBITRARILY_ROTATED_DISTANCES(0):UNROTATED_DISTANCES(0), // Each of these contains 16 4-bit distances in [0,6]
                 d1 = rotated?ARBITRARILY_ROTATED_DISTANCES(1):UNROTATED_DISTANCES(1);
  #undef DISTANCES
  // Determine minimum distance
  #define FIELDWISE_MIN(d0,d1) ({ \
    uint64_t high = each<<3; \
    uint64_t mask = ((high|d0)-(d1))&high; \
    mask |= mask>>1; \
    mask |= mask>>2; \
    (d0&~mask)|(d1&mask); })
  uint64_t d = FIELDWISE_MIN(d0,d1);
  d = FIELDWISE_MIN(d,d>>32);
  d = FIELDWISE_MIN(d,d>>16);
  d = FIELDWISE_MIN(d,d>>8);
  const int min_distance = FIELDWISE_MIN(d,d>>4)&0xf;
  // If we're in debug mode, check against the slow way
#ifndef NDEBUG
  const int slow_min_distance = min(SLOW_MIN_DISTANCE(d0,16,4),SLOW_MIN_DISTANCE(d1,16,4));
  GEODE_ASSERT(slow_min_distance==min_distance);
#endif
  // If the minimum distance is 6, a black win is impossible, so no need to count the ways
  if (min_distance==6)
    return 0;
  // Count number of times min_distance occurs
  const uint64_t mins = min_distance*each;
  #define MATCHES(d) (~((d^mins)|(d^mins)>>1|(d^mins)>>2)&each)
  const int count = popcount(MATCHES(d0)|MATCHES(d1)<<1);
  #undef MATCHES
  return ((6-min_distance)<<16)+count;
}

int unrotated_win_closeness(side_t black, side_t white) {
  return win_closeness<false>(black,white);
}

int arbitrarily_rotated_win_closeness(side_t black, side_t white) {
  return win_closeness<true>(black,white);
}

int rotated_status(board_t board) {
  check_board(board);
  return rotated_won(unpack(board,0)) ? 1 : 0;
}

// For testing purposes only: extremely slow, and checks only one side
int arbitrarily_rotated_status(board_t board) {
  check_board(board);
  const side_t side = unpack(board,0);
  quadrant_t rotated[4][4];
  for (int q=0;q<4;q++) {
    rotated[q][0] = quadrant(side,q);
    for (int i=0;i<3;i++)
      rotated[q][i+1] = rotations[rotated[q][i]][0];
  }
  for (int r0=0;r0<4;r0++) for (int r1=0;r1<4;r1++) for (int r2=0;r2<4;r2++) for (int r3=0;r3<4;r3++) {
    side_t rside = quadrants(rotated[0][r0],rotated[1][r1],rotated[2][r2],rotated[3][r3]);
    if (won(rside))
      return 1;
  }
  return 0;
}

}  // namespace pentago
