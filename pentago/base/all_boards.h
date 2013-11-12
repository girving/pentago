// Board enumeration
#pragma once

#include <pentago/base/board.h>
#include <pentago/base/section.h>
#include <geode/array/Array.h>
namespace pentago {

// Enumerate the different ways n stones can be distributed into the four quadrants
GEODE_EXPORT Array<section_t> all_boards_sections(int n, int symmetries=8);

// Print statistics about the set of n stone positions, and return the total number including redundancies
GEODE_EXPORT uint64_t all_boards_stats(int n, int symmetries);

// Enumerate all supersymmetric n stone positions, with some redundancy
GEODE_EXPORT Array<board_t> all_boards_list(int n);

// Test our enumeration
GEODE_EXPORT void all_boards_sample_test(int n, int steps);

// Given two sorted lists of boards, check that the first is contained in the second
GEODE_EXPORT bool sorted_array_is_subset(RawArray<const board_t> boards0, RawArray<const board_t> boards1);

}
