// Board definitions and utility functions

#include "gen/tables.h"
#include <cassert>
namespace pentago {

// Each board is divided into 4 quadrants, and each quadrant is stored
// in one of the 16-bit quarters of a 64-bit int.  Within a quadrant,
// the state is packed in radix 3, which works since 3**9 < 2**16.
typedef uint64_t board_t;

// A side (i.e., the set of stones occupied by one player) is similarly
// broken into 4 quadrants, but each quadrant is packed in radix 2.
typedef uint64_t side_t;

// A single quadrant always fits into uint16_t, whether in radix 2 or 3.
typedef uint16_t quadrant_t;

// Extract one quadrant from either board_t or side_t
static inline quadrant_t quadrant(uint64_t state, int q) {
  assert(0<=q && q<4);
  return (state>>16*q)&0xffff;
}

// Pack four quadrants into a single board_t or side_t
static inline uint64_t quadrants(quadrant_t q0, quadrant_t q1, quadrant_t q2, quadrant_t q3) {
  return q0|(uint64_t)q1<<16|(uint64_t)q2<<32|(uint64_t)q3<<48;
}

// Combine two side quadrants into a single double-sided quadrant
static inline quadrant_t pack(quadrant_t side0, quadrant_t side1) {
  return pack_table[side0]+2*pack_table[side1];
}

// Pack two sides into a board
static inline board_t pack(side_t side0, side_t side1) {
  return quadrants(pack(quadrant(side0,0),quadrant(side1,0)),
                   pack(quadrant(side0,1),quadrant(side1,1)),
                   pack(quadrant(side0,2),quadrant(side1,2)),
                   pack(quadrant(side0,3),quadrant(side1,3)));
}

// Extract one side from a double-sided quadrant
static inline quadrant_t unpack(quadrant_t state, int s) {
  assert(0<=s && s<2);
  return unpack_table[state][s];
}

// Extract one side of a board
static inline side_t unpack(board_t board, int s) {
  return quadrants(unpack(quadrant(board,0),s),
                   unpack(quadrant(board,1),s),
                   unpack(quadrant(board,2),s),
                   unpack(quadrant(board,3),s));
}

// Check whose turn it is (assuming black moved first)
extern bool black_to_move(board_t board);

// Throw ValueError if a board is invalid
extern void check_board(board_t board);

}
