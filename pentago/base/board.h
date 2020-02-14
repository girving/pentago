// Board definitions and utility functions
//
// For efficiency, pentago boards are packed into 64-bit integers, using
// 16 bits for each quadrant as a 9 digit radix 3 integer.  Utilities are
// provided for breaking 64-bit boards into their component quadrants, or
// switching from radix 3 to 2 (a bitmask for one side).  The code relies
// heavily on lookup tables, computed at compile time by precompute.py.
#pragma once

#include <cassert>
#include "pentago/utility/array.h"
#include "pentago/utility/popcount.h"
#include "pentago/base/gen/tables.h"
#ifndef __wasm__
#include "pentago/utility/random.h"
#include <string>
#endif
namespace pentago {

// Each board is divided into 4 quadrants, and each quadrant is stored
// in one of the 16-bit quarters of a 64-bit int.  Within a quadrant,
// the state is packed in radix 3, which works since 3**9 < 2**16.
// The two players are denoted 0 and 1, and the presence of a stone of
// player k's is denoted by 2**k.  In the interface and other slow code
// player 0 is usually black, who is assumed to move first.  During
// tree search, it is always player 0's turn (the sides are swapped
// after each move).
typedef uint64_t board_t;

// A side (i.e., the set of stones occupied by one player) is similarly
// broken into 4 quadrants, but each quadrant is packed in radix 2.
typedef uint64_t side_t;

// Bits that can be set in a valid side
const side_t side_mask = 0x01ff01ff01ff01ff;

// A single quadrant always fits into uint16_t, whether in radix 2 or 3.
typedef uint16_t quadrant_t;
const uint16_t quadrant_count = 19683; // 3**9

// Extract one quadrant from either board_t or side_t
static inline quadrant_t quadrant(uint64_t state, int q) {
  assert(0<=q && q<4);
  return (state>>16*q)&0xffff;
}

// Pack four quadrants into a single board_t or side_t
static inline uint64_t quadrants(quadrant_t q0, quadrant_t q1, quadrant_t q2, quadrant_t q3) {
  return q0|(uint64_t)q1<<16|(uint64_t)q2<<32|(uint64_t)q3<<48;
}

#ifndef __wasm__
// Combine two side quadrants into a single double-sided quadrant
static inline quadrant_t pack(quadrant_t side0, quadrant_t side1) {
  return pack_table[side0] + 2*pack_table[side1];
}

// Pack two sides into a board
static inline board_t pack(side_t side0, side_t side1) {
  return quadrants(pack(quadrant(side0, 0), quadrant(side1, 0)),
                   pack(quadrant(side0, 1), quadrant(side1, 1)),
                   pack(quadrant(side0, 2), quadrant(side1, 2)),
                   pack(quadrant(side0, 3), quadrant(side1, 3)));
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

// Extract both sides of a board
static inline Vector<side_t,2> unpack(board_t board) {
  return vec(unpack(board, 0), unpack(board, 1));
}

// Count the stones on a board
static inline int count_stones(board_t board) {
  return popcount(unpack(board, 0) | unpack(board, 1));
}

// Check whose turn it is (assuming black moved first)
bool black_to_move(board_t board);

board_t standardize(board_t board);

// Maybe swap sides
static inline board_t flip_board(board_t board, bool turn = true) {
  return pack(unpack(board, turn), unpack(board, 1-turn));
}

// Random board and side generation
side_t random_side(Random& random);
board_t random_board(Random& random);

// Generate a random board with n stones
board_t random_board(Random& random, int n);

std::string str_board(board_t board);

// Turn a board into a 6x6 grid: x-y major order, 0,0 is lower left, value 0 for empty or 2^k for player k
Array<int,2> to_table(const board_t boards);
board_t from_table(RawArray<const int,2> tables);
#endif  // !__wasm__

// Throw ValueError if a board is invalid
void check_board(board_t board);

// Slow versions for __wasm__ use
board_t slow_pack(const side_t side0, const side_t side1) __attribute__((const));
Vector<side_t,2> slow_unpack(const board_t board) __attribute__((const));
int slow_count_stones(const board_t board) __attribute__((const));
board_t slow_flip_board(const board_t board, const bool turn = true) __attribute__((const));

}
