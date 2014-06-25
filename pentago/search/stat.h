// Statistics
//
// These statistics are collected during forward tree searches to get some idea
// what's going on, since the control flow of forward tree searches is dynamic
// and complicated.  No such thing is necessary for the backwards engines, since
// all quantitative information there is knowable in advance.
#pragma once

#include <geode/structure/forward.h>
#include <stdint.h>
namespace pentago {

using namespace geode;

#define STAT(...) __VA_ARGS__

// Flip to print depth breakdown for lookups
//#define STAT_DETAIL(...) __VA_ARGS__
#define STAT_DETAIL(...)

extern uint64_t total_expanded_nodes;
extern uint64_t expanded_nodes[37];
extern uint64_t total_lookups;
extern uint64_t successful_lookups;
STAT_DETAIL(extern uint64_t lookup_detail[37], successful_lookup_detail[37];)
extern uint64_t distance_prunes;

extern Unit clear_stats();
extern Unit print_stats();

#define PRINT_STATS(bits) ({ STAT(if (!(total_expanded_nodes&((1<<bits)-1))) print_stats()); })

}
