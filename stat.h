// Statistics
#pragma once

#include <stdint.h>
namespace pentago {

#define STAT(...) __VA_ARGS__

extern uint64_t expanded_nodes;
extern uint64_t total_lookups;
extern uint64_t successful_lookups;
extern uint64_t distance_prunes;

}
