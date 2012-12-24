// Symmetry-aware board counting via Polya's enumeration theorem

#include <pentago/section.h>
#include <other/core/math/uint128.h>
namespace pentago {

uint64_t choose(int n, int k) OTHER_CONST;
uint64_t count_boards(int n, int symmetries) OTHER_CONST;

// Compute (wins,losses,total) for the given super evaluation, count each distinct locally rotated position exactly once. 
Vector<uint16_t,3> popcounts_over_stabilizers(board_t board, const Vector<super_t,2>& wins) OTHER_CONST;

// Given win/loss/total counts for each section, compute a grand total
Vector<uint64_t,3> sum_section_counts(RawArray<const section_t> sections, RawArray<const Vector<uint64_t,3>> counts);

}
