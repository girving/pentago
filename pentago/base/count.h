// Symmetry-aware board counting via Polya's enumeration theorem
//
// We'd like to count boards without explicitly listing them all.  This is easy
// if symmetries are ignored: there are exactly choose(36,n)*choose(n,n//2) n
// stone boards.  With symmetries thrown in the results are more complicated,
// since different numbers of boards have different size orbits in the symmetry
// group.  Polya's enumeration theorem solves this with an elegant application
// of generating functions.
#pragma once

#include <pentago/base/section.h>
#include <pentago/base/superscore.h>
#include <geode/math/uint128.h>
namespace pentago {

uint64_t choose(int n, int k) GEODE_CONST;
uint64_t count_boards(int n, int symmetries) GEODE_CONST;

// Compute (wins,losses,total) for the given super evaluation, count each distinct locally rotated position exactly once.
GEODE_EXPORT Vector<uint16_t,3> popcounts_over_stabilizers(board_t board, const Vector<super_t,2>& wins) GEODE_CONST;

// Given win/loss/total counts for each section, compute a grand total
Vector<uint64_t,3> sum_section_counts(RawArray<const section_t> sections, RawArray<const Vector<uint64_t,3>> counts);

}
