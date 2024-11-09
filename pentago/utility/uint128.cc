#include "pentago/utility/uint128.h"
#include "pentago/utility/format.h"
#include <iostream>
namespace pentago {

using std::cout;

string str(uint128_t n) {
  const auto lo = uint64_t(n),
             hi = uint64_t(n>>64);
  // For now, we lazily produce hexadecimal to avoid having to divide.
  return hi ? tfm::format("0x%x%016x",hi,lo) : tfm::format("0x%x",lo);
}

}

namespace std {
ostream& operator<<(ostream& output, pentago::uint128_t n) {
  return output << pentago::str(n);
}
}
