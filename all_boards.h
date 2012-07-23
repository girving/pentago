// Board enumeration
#pragma once

#include <pentago/board.h>
#include <pentago/section.h>
#include <other/core/array/Array.h>
namespace pentago {

// Enumerate the different ways n stones can be distributed into the four quadrants
Array<section_t> all_boards_sections(int n, int symmetries=8);

// Print statistics about the set of n stone positions, and return the total number including redundancies
uint64_t all_boards_stats(int n, int symmetries);

// Enumerate all supersymmetric n stone positions, with some redundancy
Array<board_t> all_boards_list(int n);

// Test our enumeration
void all_boards_sample_test(int n, int steps);

// Given two sorted lists of boards, check that the first is contained in the second
bool sorted_array_is_subset(RawArray<const board_t> boards0, RawArray<const board_t> boards1);

}
