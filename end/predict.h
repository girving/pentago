// Memory usage prediction
#pragma once

#include <stdint.h>
namespace pentago {
namespace end {

// Estimate base memory usage of compute (ignoring active lines)
uint64_t base_compute_memory_usage(const int lines);

}
}
