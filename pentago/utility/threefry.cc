#define __STDC_CONSTANT_MACROS
#include "pentago/utility/uint128.h"
#include "Random123/threefry.h"
namespace pentago {

uint128_t threefry(uint128_t key, uint128_t ctr) {
  threefry2x64_ctr_t c,k;
  uint64_t mask = -1;
  c.v[0] = ctr&mask;
  c.v[1] = (ctr>>64)&mask;
  k.v[0] = key&mask;
  k.v[1] = (key>>64)&mask;
  threefry2x64_ctr_t r = threefry2x64(c,k);
  return (uint128_t(r.v[1])<<64)|r.v[0];
}

}
