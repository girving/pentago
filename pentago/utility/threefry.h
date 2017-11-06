// Simple interface to threefry
#pragma once

#include "pentago/utility/uint128.h"
namespace pentago {

// Note that we put key first to match currying, unlike Salmon et al.
uint128_t threefry(uint128_t key, uint128_t ctr) __attribute__((const));

}
