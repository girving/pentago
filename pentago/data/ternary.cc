// Flat array of ternary values, packed 5 per byte

#include "pentago/data/ternary.h"
#include "pentago/utility/debug.h"
#include <cassert>
namespace pentago {

// pow3[i] = 3^i
static const int pow3[5] = {1, 3, 9, 27, 81};

ternaries_t::ternaries_t()
    : size(0) {}

ternaries_t::ternaries_t(const uint64_t size)
    : size(size)
    , data(CHECK_CAST_INT(ceil_div(size, uint64_t(5)))) {}

ternaries_t::~ternaries_t() {}

int ternaries_t::operator[](const uint64_t i) const {
  assert(i < size);
  return data[int(i / 5)] / pow3[i % 5] % 3;
}

void ternaries_t::set(const uint64_t i, const int v) {
  assert(i < size && unsigned(v) < 3);
  const int byte = int(i / 5);
  const int pos = int(i % 5);
  const int old = data[byte] / pow3[pos] % 3;
  data[byte] += uint8_t((v - old) * pow3[pos]);
}

}  // namespace pentago
