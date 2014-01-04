// Solve positions near the end of the game using downstream retrograde analysis
#pragma once

#include <pentago/base/superscore.h>
#include <geode/structure/Hashtable.h>
namespace pentago {

// (win,notlose)
typedef Vector<super_t,2> supers_t;

// Allocate enough memory for midsolves with at least the given number of stones
GEODE_EXPORT Array<supers_t> midsolve_workspace(const int min_slice);

// Compute the values of a board and its children, assuming the board has at least 18 stones.
GEODE_EXPORT Hashtable<board_t,supers_t> midsolve_internal(const board_t root, RawArray<supers_t> workspace);

// Compute the values of the given boards, which must be children of a single board
GEODE_EXPORT Hashtable<board_t,int> midsolve(const board_t root, const RawArray<const board_t> boards,
                                             RawArray<supers_t> workspace);

}
