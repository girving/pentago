// Random access pseudorandom permutations
#pragma once

#include "pentago/utility/uint128.h"
namespace pentago {

// Apply a pseudorandom permutation to the range [0,n-1]
uint64_t random_permute(uint64_t n, uint128_t key, uint64_t x) __attribute__((const));
uint64_t random_unpermute(uint64_t n, uint128_t key, uint64_t x) __attribute__((const));

}
