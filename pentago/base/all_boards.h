// Board enumeration
//
// Various ways of listing all boards or sections with a given number of stones,
// possibly with various numbers of symmetries eliminated.
#pragma once

#include "pentago/base/board.h"
#include "pentago/base/section.h"
#include "pentago/utility/array.h"
namespace pentago {

// Enumerate the different ways n stones can be distributed into the four quadrants
Array<section_t> all_boards_sections(int n, int symmetries=8);

// Print statistics about the set of n stone positions, and return the total number including redundancies
uint64_t all_boards_stats(int n, int symmetries);

// Enumerate all supersymmetric n stone positions, with some redundancy
Array<board_t> all_boards_list(int n);

// Enumerate all boards, reducing by the given number of symmetries
Array<board_t> all_boards(const int n, const int symmetries);

// Quadrants minimal w.r.t. rotations and reflections
Array<quadrant_t> minimal_quadrants();

}
