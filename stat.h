// Statistics
#pragma once

#include <stdint.h>
namespace pentago {

#define STAT(...) __VA_ARGS__

extern uint64_t total_expanded_nodes;
extern uint64_t expanded_nodes[37];
extern uint64_t total_lookups;
extern uint64_t successful_lookups;
extern uint64_t distance_prunes;

extern void print_stats();

#define PRINT_STATS(bits) ({ STAT(if (!(total_expanded_nodes&((1<<bits)-1))) print_stats()); })

}
