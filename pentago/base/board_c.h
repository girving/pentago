// Board definitions and utility functions
#pragma once

#include "../utility/metal.h"

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
METAL_GLOBAL side_t side_mask = 0x01ff01ff01ff01ff;

// A single quadrant always fits into uint16_t, whether in radix 2 or 3.
typedef uint16_t quadrant_t;
METAL_GLOBAL uint16_t quadrant_count = 19683; // 3**9

// Extract one quadrant from either board_t or side_t
METAL_INLINE quadrant_t quadrant(uint64_t state, int q) {
  assert(0<=q && q<4);
  return (state>>16*q)&0xffff;
}
