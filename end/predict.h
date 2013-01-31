// Memory usage prediction
#pragma once

#include <other/core/utility/config.h>
#include <stdint.h>
namespace pentago {
namespace end {

// Estimate base memory usage of compute (ignoring active lines)
OTHER_EXPORT uint64_t base_compute_memory_usage(const int lines);

}
}
