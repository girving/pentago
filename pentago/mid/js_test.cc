// Test functions for wasm / js interface

#include "pentago/mid/midengine.h"
#include "pentago/utility/wasm.h"
#include <cstdint>
namespace pentago {

// Learn about alignment
static_assert(alignof(int) == 4);
static_assert(alignof(uint64_t) == 8);
static_assert(alignof(high_board_t) == 8);
static_assert(alignof(tuple<high_board_t,int>) == 8);
static_assert(alignof(mid_values_t) == 8);

// Learn about sizes
static_assert(sizeof(int) == 4);
static_assert(sizeof(uint64_t) == 8);
static_assert(sizeof(high_board_t) == 24);
static_assert(sizeof(tuple<high_board_t,int>) == 32);
static_assert(sizeof(mid_values_t) == 8 + 32 * mid_values_t::limit);

WASM_EXPORT int sqr_test(const int n) {
  return n * n;
}

// High 32 bits of sum of 64 bit numbers
WASM_EXPORT uint32_t sum_test(const int num, const uint64_t* data) {
  uint64_t sum = 0;
  for (const auto n : RawArray<const uint64_t>(num, data))
    sum += n;
  return sum >> 32;
}

WASM_EXPORT void die_test() {
  die("An informative message");
}

}  // namespace pentago
