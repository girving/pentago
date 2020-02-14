// Score definitions and immediately evaluation functions
//
// This file became vestigial as soon as the rotation abstraction
// was introduced, since it mixes depth and values together.  This
// is incompatible with super_t, since we might need different
// depths for different bits.
#pragma once

#include "pentago/base/board.h"
namespace pentago {

// A score consists of two fields: depth and value.  The value field has two bits determining the
// the result of the game according to player 0 (assuming player 0 to play):
//     0 for loss, 1 for tie, 2 for win.
// The depth field has 7 bits giving the depth of the result.  depth >= 36 means the result is
// exact.  Losses and wins are always exact, so a depth < 36 occurs only in the case of a tie, and
// means that with perfect play, neither player can force a win or loss within depth plys.
typedef uint16_t score_t;
static const int score_bits = 9;
static const int score_mask = (1<<score_bits)-1;

// Combine depth and value fields into a score
static inline score_t score(int depth, int value) {
  return depth<<2|value;
}

// An exact (infinite depth) score
static inline score_t exact_score(int value) {
  return score(36,value);
}

// Flip a score and increment its depth by one
static inline score_t lift(score_t sc) {
  return score(1+(sc>>2),2-(sc&3));
}

// Determine if one side has 5 in a row
static inline bool won(side_t side) {
  /* To test whether a position is a win for a given player, we note that
   * there are 3*4*2+4+4 = 32 different ways of getting 5 in a row on the
   * board.  Thus, a 64-bit int can store a 2 bit field for each possible
   * method.  We then precompute a lookup table mapping each quadrant state
   * to the number of win-possibilities it contributes to.  28 of the ways
   * of winning occur between two boards, and 4 occur between 4, so a sum
   * and a few bit twiddling checks are sufficient to test whether 5 in a
   * row exists.  See precompute for more details. */
  uint64_t c = win_contributions[0][quadrant(side, 0)]
             + win_contributions[1][quadrant(side, 1)]
             + win_contributions[2][quadrant(side, 2)]
             + win_contributions[3][quadrant(side, 3)];
  return c&(c>>1)&0x55 // The first four ways of winning require contributions from three quadrants
      || c&(0xaaaaaaaaaaaaaaaa<<8); // The remaining 28 ways require contributions from only two
}

#ifndef __wasm__
// Determine if one side can win by rotating a quadrant
static inline bool rotated_won(side_t side) {
  quadrant_t q0 = quadrant(side,0),
             q1 = quadrant(side,1),
             q2 = quadrant(side,2),
             q3 = quadrant(side,3);
  // First see how far we get without rotations
  uint64_t c = win_contributions[0][q0]
             + win_contributions[1][q1]
             + win_contributions[2][q2]
             + win_contributions[3][q3];
  // We win if we only need to rotate a single quadrant
  uint64_t c0 = c+rotated_win_contribution_deltas[0][q0],
           c1 = c+rotated_win_contribution_deltas[1][q1],
           c2 = c+rotated_win_contribution_deltas[2][q2],
           c3 = c+rotated_win_contribution_deltas[3][q3];
  // Check if we won
  return (c0|c1|c2|c3)&(0xaaaaaaaaaaaaaaaa<<8) // The last remaining 28 ways of winning require contributions two quadrants
      || ((c0&(c0>>1))|(c1&(c1>>1))|(c2&(c2>>1))|(c3&(c3>>1)))&0x55; // The first four require contributions from three
}

// Evaluate the current status of a board, returning one bit for whether each player has 5 in a row.
static inline int status(board_t board) {
  const side_t side0 = unpack(board,0),
               side1 = unpack(board,1);
  const int wons = won(side0)|won(side1)<<1;
  return wons?wons // Did someone (or someones) just win?
             :(side0|side1)==0x1ff01ff01ff01ff?3:0; // If the board is full, it's an immediate tie
}

// Compute the minimum number of black moves required for a win, together with the number of different
// ways that minimum can be achieved, assuming no rotations except possibly a single rotation right at
// the end.  Returns ((6-min_distance)<<16)+count, so that a higher number means closer to a black win.
// If winning is impossible, the return value is 0.
int rotated_win_closeness(side_t black, side_t white) __attribute__((const));

// Same as above, except allowing no rotations whatsoever.
int unrotated_win_closeness(side_t black, side_t white) __attribute__((const));

// Same as above, but allow arbitrarily many rotations
int arbitrarily_rotated_win_closeness(side_t black, side_t white) __attribute__((const));

// Warning: slow, and checks only one side
int rotated_status(board_t board);
int arbitrarily_rotated_status(board_t board);
#endif  // !__wasm__

}
