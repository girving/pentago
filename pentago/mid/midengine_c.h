// Solve positions near the end of the game using downstream retrograde analysis
//
// If we fix a given 18 stone board, there are 18 free spots.  Since we abstract
// away rotations, these free spots do not move around, so the set of positions
// can be compactly indexed using combinatorial trickery.  This allows efficient
// traversal of exactly those position downstream of a given root.  For boards with
// 18 stones, this takes about 15 seconds on a Rackspace instance using less than
// 1 GB of RAM.  Conveniently, the time depends solely on the number of stones, not
// the particular board, which is perfectly suited for use as a web service.
#pragma once

#include "pentago/mid/halfsuper_c.h"
#include "pentago/high/board_c.h"

typedef struct halfsupers_t_ {
  halfsuper_s win, notlose;
} halfsupers_t;

typedef struct superinfos_t_ {
  halfsuper_s win, notlose;
  bool parity;
} superinfos_t;

typedef struct mid_super_t_ {
  side_t sides[2];
  superinfos_t supers;
} mid_super_t;
