// Move generation code
//
// This move generation code is a hideously ugly pile of macros to
// ensure loop enrolling and allocate moves entirely on the stack.
// See search/superengine.cpp for example usage.
#pragma once

#include "pentago/base/board.h"
#include "pentago/utility/integer_log.h"
#include "pentago/utility/popcount.h"
namespace pentago {

// We declare the move listing code as a huge macro in order to use it in multiple functions while taking advantage of gcc's variable size arrays.
// To use, invoke MOVES(board) to fill a board_t moves[total] array.
#define QR0(s,q,dir) rotations[quadrant(side##s,q)][dir]
#define QR1(s,q) {QR0(s,q,0),QR0(s,q,1)}
#define QR2(s) {QR1(s,0),QR1(s,1),QR1(s,2),QR1(s,3)}
#define COUNT_MOVES(q,qpp) \
  const int offset##q  = move_offsets[quadrant(filled,q)]; \
  const int count##q   = move_offsets[quadrant(filled,q)+1]-offset##q; \
  const int total##qpp = total##q + 8*count##q;
#define MOVE_QUAD(q,qr,dir,i) \
  quadrant_t side0_quad##i, side1_quad##i; \
  if (qr!=i) { \
    side0_quad##i = quadrant(side1,i); \
    side1_quad##i = q==i?changed:quadrant(side0,i); \
  } else { \
    side0_quad##i = rotated[1][i][dir]; \
    side1_quad##i = q==i?rotations[changed][dir]:rotated[0][i][dir]; \
  } \
  const quadrant_t both##i = pack(side0_quad##i,side1_quad##i);
#define MOVE(q,qr,dir) { \
  MOVE_QUAD(q,qr,dir,0) \
  MOVE_QUAD(q,qr,dir,1) \
  MOVE_QUAD(q,qr,dir,2) \
  MOVE_QUAD(q,qr,dir,3) \
  moves[total##q+8*i+2*qr+dir] = quadrants(both0,both1,both2,both3); \
}
#define COLLECT_MOVES(q) \
  for (int i=0;i<count##q;i++) { \
    const quadrant_t changed = quadrant(side0,q)|move_flat[offset##q+i]; \
    MOVE(q,0,0) MOVE(q,0,1) \
    MOVE(q,1,0) MOVE(q,1,1) \
    MOVE(q,2,0) MOVE(q,2,1) \
    MOVE(q,3,0) MOVE(q,3,1) \
  }
#define MOVES(board) \
  /* Unpack sides */ \
  const side_t side0 = unpack(board,0), \
               side1 = unpack(board,1); \
  /* Rotate all four quadrants left and right in preparation for move generation */ \
  const quadrant_t rotated[2][4][2] = {QR2(0),QR2(1)}; \
  /* Count the number of moves in each quadrant */ \
  const side_t filled = side0|side1; \
  const int total0 = 0; \
  COUNT_MOVES(0,1) COUNT_MOVES(1,2) COUNT_MOVES(2,3) COUNT_MOVES(3,4) \
  int total = total4; /* Leave mutable to allow in-place pruning of the list */ \
  /* Collect the list of all possible moves.  Note that we repack with the sides */ \
  /* flipped so that it's still player 0's turn. */ \
  board_t moves[total]; \
  COLLECT_MOVES(0) COLLECT_MOVES(1) COLLECT_MOVES(2) COLLECT_MOVES(3)

// Same as MOVES, but ignores rotations and operates in unpacked mode
#define SIMPLE_MOVES(side0,side1) \
  side_t _move_mask = side_mask^(side0|side1); \
  int total = popcount(_move_mask); \
  /* Collect the list of possible moves.  Note that only side0 changes */ \
  side_t moves[total]; \
  for (int i=0;i<total;i++) { \
    side_t move = min_bit(_move_mask); \
    moves[i] = side0 | move; \
    _move_mask ^= move; \
  }

// Simple versions for bindings
Array<board_t> moves(board_t board);
Array<board_t> simple_moves(board_t board);

}
