// Block utilities
#pragma once

#include <pentago/end/config.h>
#include <pentago/utility/ceil_div.h>
namespace pentago {
namespace end {

static inline Vector<int,4> section_blocks(section_t section) {
  return ceil_div(section.shape(),block_size);
}

template<int d> static inline Vector<int,d> block_shape(Vector<int,d> shape, Vector<uint8_t,d> block) {
  const Vector<int,d> block_(block);
  return Vector<int,d>::componentwise_min(shape,block_size*(block_+1))-block_size*block_;
}

}
}
