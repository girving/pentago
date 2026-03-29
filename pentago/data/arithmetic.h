// Arithmetic coding of ternary arrays using 8 interleaved streams
#pragma once

#include "pentago/data/ternary.h"
#include "pentago/utility/vector.h"
namespace pentago {

// Arithmetic-coded ternary data: symbol counts + compressed bytes
struct arithmetic_t {
  Vector<uint64_t,3> counts;    // measured counts of {0, 1, 2}
  Array<const uint8_t> data;    // 8-stream interleaved arithmetic-coded bytes

  uint64_t total() const { return counts[0] + counts[1] + counts[2]; }
};

// Encode: measures distribution internally
arithmetic_t arithmetic_encode(const ternaries_t data);

// Decode: uses stored counts as weights
ternaries_t arithmetic_decode(const arithmetic_t encoded);

}  // namespace pentago
