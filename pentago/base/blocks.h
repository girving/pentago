// Supertensor block utilities
#pragma once

#include "pentago/base/section.h"
#include "pentago/utility/ceil_div.h"
#include "pentago/utility/vector.h"
namespace pentago {

// Block size is fixed at 8 for all supertensor files
static constexpr int block_size = 8;

static inline Vector<int,4> section_blocks(const section_t section) {
  return ceil_div(section.shape(), block_size);
}

static inline Vector<int,4> block_shape(const Vector<int,4> shape, const Vector<uint8_t,4> block) {
  const auto b = Vector<int,4>(block);
  return cwise_min(shape, block_size * (b + 1)) - block_size * b;
}

}  // namespace pentago
