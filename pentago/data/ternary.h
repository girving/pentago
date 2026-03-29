// Flat array of ternary values, packed 5 per byte (3^5 = 243 < 256)
#pragma once

#include "pentago/utility/array.h"
#include "pentago/utility/ceil_div.h"
namespace pentago {

struct ternaries_t {
  const uint64_t size;  // number of ternary values
  const Array<uint8_t> data;  // ceil(size/5) packed bytes

  ternaries_t();
  explicit ternaries_t(const uint64_t size);
  ~ternaries_t();

  int operator[](const uint64_t i) const;
  void set(const uint64_t i, const int v);
};

}  // namespace pentago
