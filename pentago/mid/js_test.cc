// Test functions for wasm / js interface

#include "pentago/mid/midengine.h"
#include "pentago/utility/wasm.h"
#include <cstdint>
namespace pentago {

// Learn about alignment
static_assert(alignof(int) == 4);
static_assert(alignof(uint64_t) == 8);
static_assert(alignof(raw_t) == 8);
static_assert(alignof(tuple<raw_t,int>) == 8);
static_assert(alignof(mid_values_t) == 8);

// Learn about sizes
static_assert(sizeof(int) == 4);
static_assert(sizeof(uint64_t) == 8);
static_assert(sizeof(raw_t) == 8);
static_assert(sizeof(tuple<raw_t,int>) == 16);
static_assert(sizeof(mid_values_t) == 8 + 16 * mid_values_t::limit);

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

// int64_t roundtrips via BigInt
WASM_EXPORT int64_t int64_test(const int64_t n) {
  return n;
}

}  // namespace pentago
